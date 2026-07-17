"""数据加载与对话格式转换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from .prompts import SYSTEM_PROMPT


@dataclass(frozen=True)
class DatasetConfig:
    """数据集配置。

    训练和评测共用这套配置，保证两边的数据切分方式一致。
    """

    dataset_id: str = "lmms-lab/multimodal-open-r1-8k-verified"
    dataset_split: str = "train"
    test_size: int | float = 100
    seed: int = 42
    max_train_samples: int | None = None


def load_raw_dataset(cfg: DatasetConfig) -> Dataset:
    """加载原始数据集，并按需截断到较小子集。"""

    dataset = load_dataset(cfg.dataset_id, split=cfg.dataset_split)
    # 小规模调试时可以只取前 N 条，避免每次都处理完整数据集。
    if cfg.max_train_samples is not None:
        dataset = dataset.select(range(min(cfg.max_train_samples, len(dataset))))
    return dataset


def build_prompt(problem: str) -> list[dict[str, Any]]:
    """把单个题目转成 VLM 所需的消息列表。"""

    # 这里先只放文本 user message；图片保留在独立的 image 字段中。
    # 训练时 TRL/processor 会根据 prompt + image 组合成真正的图文输入。
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def to_grpo_example(example: dict[str, Any]) -> dict[str, Any]:
    """把原始样本转换成 GRPOTrainer 可直接消费的样本。"""

    # GRPOTrainer 会读取 prompt 和 image 生成 completion；
    # reward function 还需要 solution 来判断答案是否正确，所以不能删掉。
    return {
        "prompt": build_prompt(example["problem"]),
        "image": example["image"],
        "solution": example["solution"],
    }


def _count_prompt_tokens(processor: Any, prompt: list[dict[str, Any]]) -> int:
    """用训练时同一套 chat template 估算 prompt token 数。"""

    # TRL 的 GRPOConfig.max_prompt_length 已 deprecated。
    # 因此这里在数据预处理阶段主动过滤超长 prompt。
    tokenized = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    if isinstance(tokenized, dict):
        input_ids = tokenized["input_ids"]
    else:
        input_ids = tokenized

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


def prepare_datasets(
    raw_dataset: Dataset,
    processor: Any,
    *,
    test_size: int | float = 100,
    seed: int = 42,
    max_prompt_tokens: int | None = None,
) -> tuple[Dataset, Dataset]:
    """切分并转换数据集。"""

    def _filter_big_images(example: dict[str, Any]) -> bool:
        image = example["image"]
        # 图片越大，视觉 token 越多，显存压力越大。
        # Cookbook 中也会过滤较大图片，这里保持一个适合个人 GPU 的保守阈值。
        return image.size[0] < 512 and image.size[1] < 512

    def _convert_to_rgb(example: dict[str, Any]) -> dict[str, Any]:
        image = example["image"]
        # 统一成 RGB，避免灰度图、带 alpha 通道图片在 processor 中出现格式差异。
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    raw_dataset = raw_dataset.filter(_filter_big_images)
    raw_dataset = raw_dataset.map(_convert_to_rgb)

    split = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # map(remove_columns=...) 会先根据旧列名删除列。
    # keep_columns 里写的是转换后的目标列；旧数据集中没有这些列时会自然保留 map 的新输出。
    keep_columns = {"prompt", "image", "solution"}

    train_dataset = train_dataset.map(
        lambda example: to_grpo_example(example),
        remove_columns=[column for column in train_dataset.column_names if column not in keep_columns],
    )
    eval_dataset = eval_dataset.map(
        lambda example: to_grpo_example(example),
        remove_columns=[column for column in eval_dataset.column_names if column not in keep_columns],
    )

    train_dataset = _filter_overlong_prompts(train_dataset, processor, max_prompt_tokens)
    eval_dataset = _filter_overlong_prompts(eval_dataset, processor, max_prompt_tokens)

    return train_dataset, eval_dataset
