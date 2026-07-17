"""评测指标与配对比较工具。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .rewards import accuracy_reward, extract_answer_text, think_format_reward


@dataclass(frozen=True)
class ModelSpec:
    """命令行中传入的模型配置。"""

    label: str
    path: str


def parse_model_specs(raw_specs: list[str]) -> list[ModelSpec]:
    """解析 label=path 形式的模型列表。"""

    specs: list[ModelSpec] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"模型参数必须是 label=path 形式，收到：{raw}")
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"模型参数必须包含非空 label 和 path，收到：{raw}")
        specs.append(ModelSpec(label=label, path=path))
    return specs


def score_completion(completion: str, solution: str) -> dict[str, float | str]:
    """计算单条 completion 的评测指标。"""

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
    """在同一批样本上做配对比较。"""

    wins = ties = losses = 0
    reward_deltas: list[float] = []
    accuracy_deltas: list[float] = []
    format_deltas: list[float] = []

    for base, cand in zip(baseline_rows, candidate_rows):
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

