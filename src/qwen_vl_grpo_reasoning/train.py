"""GRPO 训练入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from .data import DatasetConfig, load_raw_dataset, prepare_datasets
from .rewards import accuracy_reward, think_format_reward

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """集中定义训练参数，方便 shell 脚本直接覆盖。"""

    parser = argparse.ArgumentParser(description="使用 GRPO 对 VLM 做后训练。")

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset_id", type=str, default="lmms-lab/multimodal-open-r1-8k-verified")
    parser.add_argument("--dataset_split", type=str, default="train[:5%]")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-qwen2p5-vl")
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=None,
        help="训练前过滤超过该 token 数的 prompt；用于替代 deprecated 的 GRPOConfig.max_prompt_length。",
    )

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="TensorBoard 日志目录；默认写入 output_dir/logs/tensorboard。",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="普通文本日志文件；默认写入 output_dir/logs/train.log。",
    )
    parser.add_argument(
        "--remove_unused_columns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否丢弃数据集中未被模型直接使用的列。",
    )

    parser.add_argument("--bf16", action="store_true", help="启用 bf16（推荐 GPU 环境）。")
    parser.add_argument("--fp16", action="store_true", help="启用 fp16。")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或关闭 gradient checkpointing。",
    )
    parser.add_argument(
        "--use_peft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或关闭 LoRA / PEFT。",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument("--use_vllm", action="store_true", help="使用 vLLM 加速生成（Linux 更合适）。")
    parser.add_argument("--vllm_mode", type=str, choices=["colocate", "server"], default="colocate")
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--device_map_auto", action="store_true", help="仅适合单卡/非分布式的快速实验。")

    return parser


def _setup_logging(output_dir: Path, log_file: str | None) -> Path:
    """把训练日志同时输出到终端和 output_dir 下的文件。"""

    # 每组实验都有独立 output_dir，因此日志也跟着实验目录走，后续复盘不会串。
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else logs_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 防止重复运行或 notebook 环境里重复添加 handler，导致日志打印多次。
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path


def _resolve_dtype(args: argparse.Namespace) -> Any:
    import torch

    # bf16 在 A100/H100/部分 40 系显卡上通常更稳；不支持时自动回退。
    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if args.fp16 and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _build_lora_config(args: argparse.Namespace) -> Any | None:
    if not args.use_peft:
        return None

    from peft import LoraConfig

    # target_modules 用逗号分隔，便于在 shell 脚本里直接改。
    # 默认只训练注意力中的 q_proj/v_proj，是 VLM LoRA 常见的保守起点。
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    # 先创建 output_dir 和日志文件，再做模型/数据集加载。
    # 如果下载模型或数据失败，失败原因也能写入对应实验目录。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_logging(output_dir, args.log_file)
    logging_dir = Path(args.logging_dir) if args.logging_dir else output_dir / "logs" / "tensorboard"
    logging_dir.mkdir(parents=True, exist_ok=True)

    logger.info("output_dir: %s", output_dir)
    logger.info("log_file: %s", log_path)
    logger.info("logging_dir: %s", logging_dir)
    logger.info("args: %s", args)

    # 这些依赖比较重，延迟导入可以让 `python -m ...train --help` 更轻量，
    # 也避免在只查看参数时触发本机 torch/torchvision 兼容问题。
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    # AutoProcessor 同时负责文本 chat template、tokenize，以及图像预处理。
    # 训练和评测必须尽量使用同一个 processor，避免 prompt/token 口径不一致。
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        padding_side="left",
    )

    raw_cfg = DatasetConfig(
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        test_size=args.test_size,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
    )
    raw_dataset = load_raw_dataset(raw_cfg)
    # prepare_datasets 会完成图片过滤、RGB 转换、train/eval 切分和 prompt 构造。
    train_dataset, eval_dataset = prepare_datasets(
        raw_dataset,
        processor,
        test_size=args.test_size,
        seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    dtype = _resolve_dtype(args)
    # device_map_auto 只建议单卡快速实验使用；accelerate 多卡时通常交给 accelerate 管。
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if args.device_map_auto else None,
    )

    peft_config = _build_lora_config(args)
    # GRPOConfig 里的 output_dir/logging_dir/save_steps 会决定 checkpoint 和日志落盘位置。
    # remove_unused_columns 默认 False，因为 reward function 需要 solution，VLM 还需要 image。
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=bool(args.bf16 and dtype == torch.bfloat16),
        fp16=bool(args.fp16 and dtype == torch.float16),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        logging_dir=str(logging_dir),
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        remove_unused_columns=args.remove_unused_columns,
        report_to=[args.report_to] if args.report_to else [],
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        log_completions=args.log_completions,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        # 两个 reward 会分别约束“格式正确”和“答案正确”。
        reward_funcs=[think_format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logger.info("start training")
    trainer.train(resume_from_checkpoint=)
    logger.info("training finished, saving model to %s", output_dir)
    trainer.save_model(str(output_dir))

    if args.push_to_hub:
        logger.info("pushing model to hub: %s", args.hub_model_id)
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
