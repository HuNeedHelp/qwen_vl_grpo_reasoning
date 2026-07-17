"""模型评测入口。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import HfArgumentParser

from utilities.eval_io import (
    append_jsonl,
    batched,
    build_prediction_row,
    build_run_config,
    clear_eval_outputs,
    deduplicate_rows,
    read_jsonl,
    validate_or_write_run_config,
    write_jsonl,
    write_summary_csv,
)
from utilities.eval_metrics import (
    ModelSpec,
    bootstrap_ci,
    build_paired_rows,
    paired_compare,
    parse_model_specs,
    score_completion,
    summarize_model,
)
from utilities.vlm import generate_batch, load_model


@dataclass
class EvalScriptArguments:
    """评测脚本参数。"""

    models: list[str] = field(
        metadata={
            "help": "模型列表，格式为 label=path，"
            "例如 base=Qwen/Qwen2.5-VL-3B-Instruct grpo=outputs/xxx。"
        }
    )
    processor_path: str | None = field(
        default=None,
        metadata={"help": "processor 路径；默认使用第一个模型路径。"},
    )
    dataset_id: str = field(
        default="lmms-lab/multimodal-open-r1-8k-verified",
        metadata={"help": "评测数据集 ID。"},
    )
    dataset_split: str = field(
        default="train[:5%]",
        metadata={"help": "datasets.load_dataset 使用的数据 split。"},
    )
    test_size: int = field(
        default=100,
        metadata={"help": "prepare_datasets 切分验证集时使用的验证样本数。"},
    )
    eval_samples: int = field(default=100, metadata={"help": "实际评测样本数。"})
    seed: int = field(default=42, metadata={"help": "随机种子。"})
    max_prompt_tokens: int = field(
        default=2048,
        metadata={"help": "评测前过滤超过该 token 数的 prompt。"},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "每条样本最多生成的新 token 数。"},
    )
    eval_batch_size: int = field(default=1, metadata={"help": "批量推理 batch size。"})
    temperature: float = field(default=0.0, metadata={"help": "生成 temperature；0 表示贪心生成。"})
    top_p: float = field(default=1.0, metadata={"help": "采样 top_p。"})
    bootstrap_samples: int = field(
        default=1000,
        metadata={"help": "bootstrap 置信区间采样次数。"},
    )
    output_dir: str = field(default="outputs/eval", metadata={"help": "评测结果输出目录。"})
    resume: bool = field(default=True, metadata={"help": "是否从已有 predictions.jsonl 断点续评。"})
    disable_tqdm: bool = field(default=False, metadata={"help": "关闭 tqdm 进度条。"})
    device_map: str = field(
        default="auto",
        metadata={"help": "模型加载时传给 from_pretrained 的 device_map。"},
    )


def build_arg_parser() -> HfArgumentParser:
    """返回 Hugging Face dataclass 参数解析器，兼容旧测试里的 parse_args 调用。"""

    return HfArgumentParser(EvalScriptArguments, description="评测 base model 和 GRPO 后训练模型。")


def parse_args(argv: list[str] | None = None) -> EvalScriptArguments:
    """解析命令行参数为 dataclass 对象。"""

    return build_arg_parser().parse_args_into_dataclasses(argv)[0]


def _prepare_eval_dataset(args: EvalScriptArguments, processor: Any) -> Any:
    from transformers import AutoProcessor  # noqa: F401 - 让静态分析知道 processor 类型来自 transformers

    from utilities.data import DatasetConfig, load_raw_dataset, prepare_datasets

    raw_cfg = DatasetConfig(
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        test_size=args.test_size,
        seed=args.seed,
    )
    raw_dataset = load_raw_dataset(raw_cfg)
    _, eval_dataset = prepare_datasets(
        raw_dataset,
        processor,
        test_size=args.test_size,
        seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    return eval_dataset.select(range(min(args.eval_samples, len(eval_dataset))))


def _evaluate_model(
    *,
    spec: ModelSpec,
    eval_dataset: Any,
    output_dir: Path,
    args: EvalScriptArguments,
    processor: Any,
    tqdm: Any,
) -> list[dict[str, Any]]:
    predictions_path = output_dir / f"{spec.label}_predictions.jsonl"
    if args.resume:
        rows = deduplicate_rows(read_jsonl(predictions_path), max_sample_id=len(eval_dataset))
        if predictions_path.exists():
            write_jsonl(predictions_path, rows)
    else:
        rows = []
        predictions_path.write_text("", encoding="utf-8")

    completed_ids = {int(row["sample_id"]) for row in rows}
    pending_ids = [idx for idx in range(len(eval_dataset)) if idx not in completed_ids]

    print(
        f"Evaluating {spec.label}: {spec.path} "
        f"(done={len(completed_ids)}, pending={len(pending_ids)}, batch={args.eval_batch_size})",
        flush=True,
    )

    if not pending_ids:
        return rows

    import torch

    model = load_model(spec.path, device_map=args.device_map)
    progress_bar = tqdm(
        total=len(eval_dataset),
        initial=len(completed_ids),
        desc=f"eval:{spec.label}",
        dynamic_ncols=True,
        unit="sample",
        disable=args.disable_tqdm,
    )

    for sample_ids in batched(pending_ids, args.eval_batch_size):
        batch_examples = [eval_dataset[idx] for idx in sample_ids]
        batch_outputs = generate_batch(
            model,
            processor,
            batch_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        new_rows = [
            build_prediction_row(
                sample_id=sample_id,
                model_label=spec.label,
                example=example,
                completion=completion,
                completion_tokens=completion_tokens,
                latency=latency,
            )
            for sample_id, example, (completion, completion_tokens, latency) in zip(
                sample_ids,
                batch_examples,
                batch_outputs,
            )
        ]
        rows.extend(new_rows)
        append_jsonl(predictions_path, new_rows)
        progress_bar.update(len(new_rows))

        last_row = new_rows[-1]
        progress_bar.set_postfix(
            {
                "done": len(rows),
                "format": f"{float(last_row['format_score']):.0f}",
                "acc": f"{float(last_row['accuracy']):.0f}",
                "reward": f"{float(last_row['total_reward']):.1f}",
                "latency": f"{float(last_row['latency_seconds']):.2f}s",
            }
        )

    progress_bar.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return deduplicate_rows(rows, max_sample_id=len(eval_dataset))


def main() -> None:
    args = parse_args()
    specs = parse_model_specs(args.models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor_path = args.processor_path or specs[0].path
    run_config = build_run_config(specs, processor_path, args)
    if not args.resume:
        clear_eval_outputs(output_dir, specs)
    validate_or_write_run_config(
        output_dir / "run_config.json",
        run_config,
        resume=args.resume,
        has_predictions=any((output_dir / f"{spec.label}_predictions.jsonl").exists() for spec in specs),
    )

    from tqdm.auto import tqdm
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_path, use_fast=True, padding_side="left")
    eval_dataset = _prepare_eval_dataset(args, processor)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "dataset_id": args.dataset_id,
        "dataset_split": args.dataset_split,
        "eval_samples": len(eval_dataset),
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "resume": args.resume,
        "models": {},
        "paired_comparisons": {},
    }

    for spec in specs:
        rows = _evaluate_model(
            spec=spec,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            args=args,
            processor=processor,
            tqdm=tqdm,
        )
        all_rows[spec.label] = rows
        summary["models"][spec.label] = summarize_model(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_jsonl(output_dir / f"{spec.label}_predictions.jsonl", rows)

    baseline_label = specs[0].label
    for spec in specs[1:]:
        comparison_name = f"{spec.label}_vs_{baseline_label}"
        baseline_rows = all_rows[baseline_label]
        candidate_rows = all_rows[spec.label]
        summary["paired_comparisons"][comparison_name] = paired_compare(
            baseline_rows,
            candidate_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100,
        )
        write_jsonl(
            output_dir / f"{comparison_name}_paired.jsonl",
            build_paired_rows(baseline_label, baseline_rows, spec.label, candidate_rows),
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_csv(output_dir / "summary.csv", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = [
    "bootstrap_ci",
    "build_arg_parser",
    "build_paired_rows",
    "EvalScriptArguments",
    "paired_compare",
    "parse_model_specs",
    "parse_args",
    "score_completion",
]


if __name__ == "__main__":
    main()
