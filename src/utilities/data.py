"""数据加载与对话格式转换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .prompts import SYSTEM_PROMPT


@dataclass(frozen=True)
class DatasetConfig:
    """数据集配置。"""

    dataset_id: str = "lmms-lab/multimodal-open-r1-8k-verified"
    train_size: int | float | None = None
    test_size: int | float = 100
    seed: int = 42


def load_raw_dataset(cfg: DatasetConfig) -> Dataset:
    """加载 Hugging Face Hub 上完整的 train split。"""

    return load_dataset(cfg.dataset_id, split="train")


def build_prompt(problem: str) -> list[dict[str, Any]]:
    """把单个题目转成 VLM 所需的消息列表。"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def to_grpo_example(example: dict[str, Any]) -> dict[str, Any]:
    """把原始样本转换成 GRPOTrainer 可直接消费的样本。"""

    return {
        "prompt": build_prompt(example["problem"]),
        "image": example["image"],
        "solution": example["solution"],
    }


def _count_prompt_tokens(processor: Any, prompt: list[dict[str, Any]]) -> int:
    """用训练时同一套 chat template 估算 prompt token 数。"""

    tokenized = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    input_ids = tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized

    if hasattr(input_ids, "shape"):
        shape = input_ids.shape
        return int(shape[-1]) if len(shape) > 0 else 0

    if len(input_ids) > 0 and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _filter_overlong_prompts(dataset: Dataset, processor: Any, max_prompt_tokens: int | None) -> Dataset:
    """过滤超过指定 token 数的 prompt，替代 deprecated 的 max_prompt_length。"""

    if max_prompt_tokens is None:
        return dataset

    return dataset.filter(
        lambda example: _count_prompt_tokens(processor, example["prompt"]) <= max_prompt_tokens,
        desc=f"Filter prompts longer than {max_prompt_tokens} tokens",
    )


def preprocess_raw_dataset(raw_dataset: Dataset) -> Dataset:
    """过滤过大图片并统一转成 RGB，保证后续切分前的数据池一致。"""

    def _filter_big_images(example: dict[str, Any]) -> bool:
        image = example["image"]
        return image.size[0] < 512 and image.size[1] < 512

    def _convert_to_rgb(example: dict[str, Any]) -> dict[str, Any]:
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    raw_dataset = raw_dataset.filter(_filter_big_images)
    return raw_dataset.map(_convert_to_rgb)


def split_raw_dataset(
    raw_dataset: Dataset,
    *,
    train_size: int | float | None = None,
    test_size: int | float = 100,
    seed: int = 42,
) -> DatasetDict:
    """在同一个原始数据池上做确定性 train/eval 切分。"""

    raw_dataset = preprocess_raw_dataset(raw_dataset)
    split_kwargs: dict[str, int | float] = {"test_size": test_size}
    if train_size is not None:
        split_kwargs["train_size"] = train_size
    return raw_dataset.train_test_split(seed=seed, **split_kwargs)


def format_and_filter_grpo_dataset(dataset: Dataset, processor: Any, max_prompt_tokens: int | None = None) -> Dataset:
    """把原始样本转换成 GRPO 样本，并按需过滤过长 prompt。"""

    keep_columns = {"prompt", "image", "solution"}
    dataset = dataset.map(
        lambda example: to_grpo_example(example),
        remove_columns=[column for column in dataset.column_names if column not in keep_columns],
    )
    return _filter_overlong_prompts(dataset, processor, max_prompt_tokens)


def prepare_datasets(
    raw_dataset: Dataset,
    processor: Any,
    *,
    train_size: int | float | None = None,
    test_size: int | float = 100,
    eval_samples: int | None = None,
    seed: int = 42,
    max_prompt_tokens: int | None = None,
) -> tuple[Dataset, Dataset]:
    """先在原始数据池上切分，再分别转换成 GRPOTrainer 可用格式。"""
    # 拆分数据集
    split = split_raw_dataset(raw_dataset, train_size=train_size, test_size=test_size, seed=seed)
    train_split, test_split = split["train"], split["test"]
    # 选取部分验证集，避免评测整个验证集
    if eval_samples is not None:
        test_split = test_split.select(range(min(eval_samples, len(test_split))))

    train_dataset = format_and_filter_grpo_dataset(train_split, processor, max_prompt_tokens)
    eval_dataset = format_and_filter_grpo_dataset(test_split, processor, max_prompt_tokens)

    return train_dataset, eval_dataset
