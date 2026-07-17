"""GRPO 奖励函数。"""

from __future__ import annotations

import re
from typing import Any

_FORMAT_RE = re.compile(r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>$", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
except Exception:  # pragma: no cover - optional dependency fallback
    LatexExtractionConfig = None  # type: ignore[assignment]
    NormalizationConfig = None  # type: ignore[assignment]
    parse = None  # type: ignore[assignment]
    verify = None  # type: ignore[assignment]


def _flatten_completion(completion: Any) -> str:
    """兼容不同 trainer/processor 可能传入的 completion 结构。"""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            parts.append(str(item.get("content", "")) if isinstance(item, dict) else str(item))
        return "".join(parts)
    return str(completion)


def extract_answer_text(completion: Any) -> str:
    """从 `<answer>...</answer>` 中提取最终答案。"""

    text = _flatten_completion(completion).strip()
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def think_format_reward(completions: list[Any], **_: Any) -> list[float]:
    """格式奖励：鼓励模型严格输出 think + answer 结构。"""

    rewards: list[float] = []
    for completion in completions:
        text = _flatten_completion(completion).strip()
        rewards.append(1.0 if _FORMAT_RE.match(text) else 0.0)
    return rewards


def _normalize_text(text: str) -> str:
    """做一个稳妥的文本归一化，方便 fallback 比较。"""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("：", ":").replace("，", ",").replace("。", ".")
    return text


def _math_verify_reward(prediction: str, solution: str) -> float | None:
    """尽量使用 math_verify 做数学答案校验；失败时返回 None。"""

    if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
        return None

    try:
        gold_parsed = parse(solution, extraction_mode="first_match")
    except Exception:
        return None

    if not gold_parsed:
        return None

    try:
        pred_parsed = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        return float(verify(gold_parsed, pred_parsed))
    except Exception:
        return None


def accuracy_reward(
    completions: list[Any],
    solution: list[str] | None = None,
    ground_truth: list[str] | None = None,
    **kwargs: Any,
) -> list[float | None]:
    """答案奖励：优先做数学验证，失败后退回到规范化字符串比较。"""

    targets = solution or ground_truth or kwargs.get("solution") or kwargs.get("ground_truth")
    if targets is None:
        raise ValueError("accuracy_reward 需要 solution 或 ground_truth 列。")

    rewards: list[float | None] = []
    for completion, target in zip(completions, targets):
        prediction = extract_answer_text(completion)
        reward = _math_verify_reward(prediction, target)
        if reward is None:
            reward = 1.0 if _normalize_text(prediction) == _normalize_text(target) else 0.0
        rewards.append(reward)
    return rewards


__all__ = ["accuracy_reward", "extract_answer_text", "think_format_reward"]
