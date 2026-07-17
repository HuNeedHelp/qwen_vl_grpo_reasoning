"""评测文件读写、断点续评和结果表格工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .eval_metrics import ModelSpec, score_completion


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """原子保存 JSONL，避免覆盖过程中断导致原文件损坏。"""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """追加写入 JSONL，用于断点续评。"""

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取已有 JSONL，并容忍最后一行被截断。"""

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


def build_run_config(specs: list[ModelSpec], processor_path: str, args: Any) -> dict[str, Any]:
    """构造评测配置指纹。"""

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
