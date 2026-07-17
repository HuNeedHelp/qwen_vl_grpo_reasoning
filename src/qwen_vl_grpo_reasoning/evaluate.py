"""模型评测入口。

这个脚本用于在同一份留出集上对比 base model 和 GRPO 后训练模型。
它会输出逐样本预测、整体指标、bootstrap 置信区间和配对比较结果，方便写进 README 或简历。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rewards import accuracy_reward, extract_answer_text, think_format_reward


@dataclass(frozen=True)
class ModelSpec:
    """命令行中传入的模型配置。"""

    label: str
    path: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测 base model 和 GRPO 后训练模型。")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help='模型列表，格式为 label=path，例如 base=Qwen/Qwen2.5-VL-3B-Instruct grpo=outputs/xxx。',
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="processor 路径；默认使用第一个模型路径。",
    )
    parser.add_argument("--dataset_id", type=str, default="lmms-lab/multimodal-open-r1-8k-verified")
    parser.add_argument("--dataset_split", type=str, default="train[:5%]")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="评测推理 batch size；显存足够时可以调大到 2/4 提升吞吐。",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="从已有 predictions.jsonl 断点续评；默认开启。",
    )
    parser.add_argument(
        "--no_resume",
        "--no-resume",
        dest="resume",
        action="store_false",
        help="忽略已有 predictions.jsonl，强制重新评测。",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="关闭评测进度条，适合某些日志系统或后台任务。",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='模型加载的 device_map，默认 "auto"；显存不足时通常保持默认即可。',
    )
    return parser


def parse_model_specs(raw_specs: list[str]) -> list[ModelSpec]:
    """解析 label=path 形式的模型列表。"""

    specs: list[ModelSpec] = []
    for raw in raw_specs:
        # label 会用于输出文件名，例如 base_predictions.jsonl。
        if "=" not in raw:
            raise ValueError(f"模型参数必须是 label=path 形式，收到：{raw}")
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"模型参数必须包含非空 label 和 path，收到：{raw}")
        specs.append(ModelSpec(label=label, path=path))
    return specs


def load_model(model_path: str, *, device_map: str = "auto") -> Any:
    """加载普通模型或 PEFT/LoRA adapter。

    如果 model_path 是 LoRA adapter 目录，PEFT 会从 adapter_config.json 中读取 base model。
    这样可以同时评测 base model 和 trainer.save_model 保存出来的 adapter。
    """

    from peft import PeftConfig, PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_dir = Path(model_path)
    adapter_config = model_dir / "adapter_config.json"

    # LoRA/PEFT 训练通常只保存 adapter。检测到 adapter_config.json 时，
    # 先恢复 base model，再把 adapter 挂上去评测。
    if adapter_config.exists():
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype="auto",
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
        )

    model.eval()
    return model


def add_image_to_prompt(prompt: list[dict[str, Any]], image: Any) -> list[dict[str, Any]]:
    """把数据集中的 text-only user prompt 转成 Qwen-VL 推理需要的图文 prompt。"""

    conversation: list[dict[str, Any]] = []
    for message in prompt:
        if message["role"] != "user":
            conversation.append(message)
            continue

        # 训练数据里 image 是独立列；推理时 Qwen-VL 需要把图片塞回 user content。
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": message["content"]},
                ],
            }
        )
    return conversation


def generate_one(
    model: Any,
    processor: Any,
    prompt: list[dict[str, Any]],
    image: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int, float]:
    """对单个图文样本做生成，并返回 completion、生成 token 数和耗时。"""

    return generate_batch(
        model,
        processor,
        [{"prompt": prompt, "image": image}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0]


def generate_batch(
    model: Any,
    processor: Any,
    examples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, int, float]]:
    """批量生成 completion。

    Qwen-VL 的 processor 支持把多条图文 prompt 一次性 padding 成 batch。
    这里把 batch 总耗时平均分摊给每个样本，用于估计单样本延迟。
    """

    import torch
    from qwen_vl_utils import process_vision_info

    conversations = [add_image_to_prompt(example["prompt"], example["image"]) for example in examples]
    # 先用 chat template 得到带特殊标记的文本 prompt，再交给 processor tokenize。
    prompt_texts = [
        processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in conversations
    ]
    # process_vision_info 会从 conversation 中提取图片，并做 Qwen-VL 需要的视觉预处理。
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
        text=prompt_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    # temperature=0 时使用确定性 greedy decoding，不传 temperature/top_p，避免 transformers 警告。
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    start = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    latency = time.perf_counter() - start

    completion_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    # batch_decode 对 Python list 更稳，也避免某些版本在 GPU tensor 上行为不一致。
    completion_token_lists = [ids.detach().cpu().tolist() for ids in completion_ids]
    completions = processor.batch_decode(
        completion_token_lists,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    latency_per_sample = latency / max(len(examples), 1)
    return [
        (completion, len(ids), latency_per_sample)
        for completion, ids in zip(completions, completion_token_lists)
    ]


def score_completion(completion: str, solution: str) -> dict[str, float | str]:
    """计算单条 completion 的评测指标。"""

    # 评测指标和训练 reward 保持同源，方便判断训练目标是否迁移到留出集。
    format_score = think_format_reward([completion])[0]
    accuracy = accuracy_reward([completion], solution=[solution])[0]
    accuracy_value = 0.0 if accuracy is None else float(accuracy)
    total_reward = format_score + accuracy_value
    answer_text = extract_answer_text(completion)

    return {
        "format_score": float(format_score),
        "accuracy": accuracy_value,
        "total_reward": float(total_reward),
        "answer": answer_text,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> dict[str, float]:
    """用 bootstrap 估计均值的 95% 置信区间。"""

    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    # 对样本做有放回重采样，得到均值分布，再取 2.5% / 97.5% 分位点。
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        boot_means.append(mean(resample))

    boot_means.sort()
    low_idx = int(0.025 * (len(boot_means) - 1))
    high_idx = int(0.975 * (len(boot_means) - 1))
    return {
        "mean": mean(values),
        "ci_low": boot_means[low_idx],
        "ci_high": boot_means[high_idx],
    }


def summarize_model(rows: list[dict[str, Any]], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    """汇总单个模型的评测指标。"""

    format_values = [float(row["format_score"]) for row in rows]
    accuracy_values = [float(row["accuracy"]) for row in rows]
    reward_values = [float(row["total_reward"]) for row in rows]
    length_values = [float(row["completion_tokens"]) for row in rows]
    latency_values = [float(row["latency_seconds"]) for row in rows]

    return {
        "num_samples": len(rows),
        "format_rate": bootstrap_ci(format_values, samples=bootstrap_samples, seed=seed),
        "accuracy": bootstrap_ci(accuracy_values, samples=bootstrap_samples, seed=seed + 1),
        "avg_total_reward": bootstrap_ci(reward_values, samples=bootstrap_samples, seed=seed + 2),
        "avg_completion_tokens": mean(length_values),
        "avg_latency_seconds": mean(latency_values),
        "invalid_output_rate": 1.0 - mean(format_values),
    }


def paired_compare(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """在同一批样本上做配对比较，比单独均值更能说明模型改进是否稳定。"""

    wins = ties = losses = 0
    reward_deltas: list[float] = []
    accuracy_deltas: list[float] = []
    format_deltas: list[float] = []

    for base, cand in zip(baseline_rows, candidate_rows):
        # 同一 sample_id 上比较 candidate 和 baseline，比两个独立均值更能说明改进是否稳定。
        delta = float(cand["total_reward"]) - float(base["total_reward"])
        reward_deltas.append(delta)
        accuracy_deltas.append(float(cand["accuracy"]) - float(base["accuracy"]))
        format_deltas.append(float(cand["format_score"]) - float(base["format_score"]))

        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1

    total = len(reward_deltas) or 1
    return {
        "num_samples": len(reward_deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "loss_rate": losses / total,
        "total_reward_delta": bootstrap_ci(reward_deltas, samples=bootstrap_samples, seed=seed),
        "accuracy_delta": bootstrap_ci(accuracy_deltas, samples=bootstrap_samples, seed=seed + 1),
        "format_delta": bootstrap_ci(format_deltas, samples=bootstrap_samples, seed=seed + 2),
    }


def build_paired_rows(
    baseline_label: str,
    baseline_rows: list[dict[str, Any]],
    candidate_label: str,
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """生成逐题配对明细，方便定位 GRPO 到底改善或伤害了哪些样本。"""

    paired_rows: list[dict[str, Any]] = []
    for base, cand in zip(baseline_rows, candidate_rows):
        reward_delta = float(cand["total_reward"]) - float(base["total_reward"])
        accuracy_delta = float(cand["accuracy"]) - float(base["accuracy"])
        format_delta = float(cand["format_score"]) - float(base["format_score"])
        if reward_delta > 0:
            outcome = "win"
        elif reward_delta < 0:
            outcome = "loss"
        else:
            outcome = "tie"

        paired_rows.append(
            {
                "sample_id": base["sample_id"],
                "baseline_model": baseline_label,
                "candidate_model": candidate_label,
                "outcome": outcome,
                "reward_delta": reward_delta,
                "accuracy_delta": accuracy_delta,
                "format_delta": format_delta,
                "question": base["question"],
                "solution": base["solution"],
                "baseline_completion": base["completion"],
                "candidate_completion": cand["completion"],
                "baseline_total_reward": base["total_reward"],
                "candidate_total_reward": cand["total_reward"],
                "baseline_accuracy": base["accuracy"],
                "candidate_accuracy": cand["accuracy"],
                "baseline_format_score": base["format_score"],
                "candidate_format_score": cand["format_score"],
            }
        )
    return paired_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """原子保存 JSONL，避免覆盖过程中断导致原文件损坏。"""

    # 先写临时文件，再 replace。这样即使覆盖时中断，旧文件也不会只剩半截。
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """追加写入 JSONL，用于断点续评。"""

    # 每个 batch 完成后立即 append，断电时最多损失当前正在生成的 batch。
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取已有 JSONL。

    如果进程恰好在写最后一行时中断，最后一行可能是不完整 JSON；这种情况会忽略该行，
    后续主流程会重新生成对应样本。中间行损坏仍然报错，避免静默丢失结果。
    """

    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    last_nonempty_index = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == last_nonempty_index:
                break
            raise ValueError(f"{path} 第 {index + 1} 行不是合法 JSON，无法安全续评。") from exc
    return rows


def validate_or_write_run_config(
    path: Path,
    current_config: dict[str, Any],
    *,
    resume: bool,
    has_predictions: bool,
) -> None:
    """校验续评配置，避免把不同实验的预测混在一起。"""

    if resume and path.exists():
        # 续评时必须保证模型、数据集、seed、生成参数都一致。
        # 否则旧 predictions 和新 predictions 混合后，summary 会失去可信度。
        with path.open("r", encoding="utf-8") as f:
            previous_config = json.load(f)
        if previous_config != current_config:
            changed_keys = sorted(
                key
                for key in set(previous_config) | set(current_config)
                if previous_config.get(key) != current_config.get(key)
            )
            raise ValueError(
                "检测到续评配置发生变化："
                f"{', '.join(changed_keys)}。请更换 output_dir，或使用 --no_resume 重新评测。"
            )
        return

    if resume and has_predictions and not path.exists():
        raise ValueError(
            "发现已有 predictions.jsonl，但缺少 run_config.json，无法确认实验配置是否一致。"
            "请更换 output_dir，或使用 --no_resume 重新评测。"
        )

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(current_config, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def build_run_config(specs: list[ModelSpec], processor_path: str, args: argparse.Namespace) -> dict[str, Any]:
    """构造评测配置指纹。

    这个配置会在加载模型和数据集之前写入磁盘。这样即使首次运行在下载数据或加载模型时中断，
    下次运行也能知道当前 output_dir 对应的是哪一次实验。
    """

    return {
        "models": [{"label": spec.label, "path": spec.path} for spec in specs],
        "processor_path": processor_path,
        "dataset_id": args.dataset_id,
        "dataset_split": args.dataset_split,
        "test_size": args.test_size,
        "eval_samples": args.eval_samples,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device_map": args.device_map,
    }


def clear_eval_outputs(output_dir: Path, specs: list[ModelSpec]) -> None:
    """强制重跑前清理本脚本会生成的评测文件。"""

    # 只清理评测脚本自己生成的文件，避免误删用户放在 output_dir 下的其他材料。
    labels = [spec.label for spec in specs]
    paths = [
        output_dir / "run_config.json",
        output_dir / "summary.json",
        output_dir / "summary.csv",
    ]
    paths.extend(output_dir / f"{label}_predictions.jsonl" for label in labels)
    for candidate in labels[1:]:
        paths.append(output_dir / f"{candidate}_vs_{labels[0]}_paired.jsonl")

    for path in paths:
        path.unlink(missing_ok=True)


def deduplicate_rows(rows: list[dict[str, Any]], *, max_sample_id: int) -> list[dict[str, Any]]:
    """按 sample_id 去重并排序，续评时保留最后一次写入的结果。"""

    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, int):
            continue
        if 0 <= sample_id < max_sample_id:
            by_id[sample_id] = row
    return [by_id[sample_id] for sample_id in sorted(by_id)]


def batched(values: list[int], batch_size: int) -> list[list[int]]:
    """把 sample_id 切成固定大小的小批次。"""

    if batch_size < 1:
        raise ValueError("eval_batch_size 必须大于等于 1。")
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def build_prediction_row(
    *,
    sample_id: int,
    model_label: str,
    example: dict[str, Any],
    completion: str,
    completion_tokens: int,
    latency: float,
) -> dict[str, Any]:
    """把单条生成结果整理成稳定的 JSONL schema。"""

    scores = score_completion(completion, example["solution"])
    user_question = example["prompt"][-1]["content"]
    return {
        "sample_id": sample_id,
        "model": model_label,
        "question": user_question,
        "solution": example["solution"],
        "completion": completion,
        "completion_tokens": completion_tokens,
        "latency_seconds": latency,
        **scores,
    }


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    """保存一份适合贴到 README/简历里的扁平 CSV。"""

    fieldnames = [
        "model",
        "num_samples",
        "format_rate",
        "format_ci_low",
        "format_ci_high",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "avg_total_reward",
        "reward_ci_low",
        "reward_ci_high",
        "invalid_output_rate",
        "avg_completion_tokens",
        "avg_latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in summary["models"].items():
            writer.writerow(
                {
                    "model": label,
                    "num_samples": metrics["num_samples"],
                    "format_rate": metrics["format_rate"]["mean"],
                    "format_ci_low": metrics["format_rate"]["ci_low"],
                    "format_ci_high": metrics["format_rate"]["ci_high"],
                    "accuracy": metrics["accuracy"]["mean"],
                    "accuracy_ci_low": metrics["accuracy"]["ci_low"],
                    "accuracy_ci_high": metrics["accuracy"]["ci_high"],
                    "avg_total_reward": metrics["avg_total_reward"]["mean"],
                    "reward_ci_low": metrics["avg_total_reward"]["ci_low"],
                    "reward_ci_high": metrics["avg_total_reward"]["ci_high"],
                    "invalid_output_rate": metrics["invalid_output_rate"],
                    "avg_completion_tokens": metrics["avg_completion_tokens"],
                    "avg_latency_seconds": metrics["avg_latency_seconds"],
                }
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    specs = parse_model_specs(args.models)

    # run_config 在任何重依赖加载之前写入，保证首次运行即使中途失败，也能留下实验配置。
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

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoProcessor

    from .data import DatasetConfig, load_raw_dataset, prepare_datasets

    processor = AutoProcessor.from_pretrained(processor_path, use_fast=True, padding_side="left")

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
    eval_dataset = eval_dataset.select(range(min(args.eval_samples, len(eval_dataset))))

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
        predictions_path = output_dir / f"{spec.label}_predictions.jsonl"
        if args.resume:
            # 已完成样本保存在 predictions JSONL 中；按 sample_id 去重后即可恢复进度。
            rows = deduplicate_rows(read_jsonl(predictions_path), max_sample_id=len(eval_dataset))
            if predictions_path.exists():
                # 顺手整理文件：去掉重复行、截断行和非法 sample_id。
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

        if pending_ids:
            model = load_model(spec.path, device_map=args.device_map)
            pending_batches = batched(pending_ids, args.eval_batch_size)
            progress_bar = tqdm(
                total=len(eval_dataset),
                initial=len(completed_ids),
                desc=f"eval:{spec.label}",
                dynamic_ncols=True,
                unit="sample",
                disable=args.disable_tqdm,
            )
            for sample_ids in pending_batches:
                # datasets.Dataset 支持按 index 取单条；这里按 sample_id 组 batch。
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
                    # 统一整理 JSONL schema，后续 summary 和 paired comparison 都依赖这些字段。
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

        rows = deduplicate_rows(rows, max_sample_id=len(eval_dataset))
        all_rows[spec.label] = rows
        summary["models"][spec.label] = summarize_model(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_jsonl(output_dir / f"{spec.label}_predictions.jsonl", rows)

    baseline_label = specs[0].label
    for spec in specs[1:]:
        # 默认第一个模型是 baseline，后续模型都和它做同题配对比较。
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


if __name__ == "__main__":
    main()
