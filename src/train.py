"""GRPO 训练入口。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from re import U
from typing import Optional
from pathlib import Path
from typing import Any, Union, Optional

from loguru import logger
from transformers import HfArgumentParser

from utilities.data import DatasetConfig, load_raw_dataset, prepare_datasets
from utilities.rewards import accuracy_reward, think_format_reward


@dataclass
class TrainScriptArguments:
    """训练脚本参数。

    HfArgumentParser 会读取 dataclass 字段，自动生成命令行参数和 --help。
    """

    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        metadata={"help": "基础 VLM 模型路径或 Hugging Face Hub ID。"},
    )
    dataset_id: str = field(
        default="lmms-lab/multimodal-open-r1-8k-verified",
        metadata={"help": "训练数据集 ID。"},
    )
    train_size: Optional[float] = field(
        default=None,
        metadata={
            "help": "本地 train_test_split 后使用的训练样本数。传入整数值（如 500）表示样本数，传入浮点数（如 0.8）表示比例；None 表示使用除 test_size 外的全部样本。"
        },
    )
    output_dir: str = field(
        default="outputs/grpo-qwen2p5-vl",
        metadata={"help": "模型、checkpoint 和日志输出目录。"},
    )
    test_size: float = field(default=100, metadata={"help": "本地 train_test_split 切出的验证集大小。整数表示样本数，浮点数表示比例（如 0.2）。"})
    eval_samples: Optional[int] = field(default=None, metadata={"help": "实际评测样本数。"})
    seed: int = field(default=42, metadata={"help": "随机种子。"})
    max_prompt_tokens: int | None = field(
        default=None,
        metadata={"help": "训练前过滤超过该 token 数的 prompt；用于替代 deprecated 的 GRPOConfig.max_prompt_length。"},
    )

    learning_rate: float = field(default=1e-5, metadata={"help": "GRPO 学习率。"})
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "训练 epoch 数；max_steps > 0 时通常由 max_steps 控制。"},
    )
    max_steps: int = field(default=-1, metadata={"help": "最大训练步数；-1 表示按 epoch 训练。"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "单卡 batch size。"})
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数。"},
    )
    num_generations: int = field(default=2, metadata={"help": "每个 prompt 采样多少个 completion。"})
    max_completion_length: int = field(
        default=1024,
        metadata={"help": "生成 completion 的最大 token 数。"},
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "训练中评估策略，例如 no/steps/epoch。", "choices": ["no", "steps", "epoch"]},
    )
    eval_steps: int | None = field(
        default=50,
        metadata={"help": "eval_strategy=steps 时，每隔多少个 step 评估一次。"},
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "单卡评估 batch size。"},
    )
    logging_steps: int = field(default=10, metadata={"help": "训练日志记录间隔。"})
    save_steps: int = field(default=50, metadata={"help": "checkpoint 保存间隔。"})
    save_strategy: str = field(default="steps", metadata={"help": "保存策略，例如 steps/no/epoch。"})
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "Trainer 上报目标；为空字符串表示关闭。"},
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": '从 checkpoint 续训；可传具体目录，也可传 "last" 自动使用 output_dir 下最新 checkpoint。'},
    )
    logging_dir: str | None = field(
        default=None,
        metadata={"help": "TensorBoard 日志目录；默认写入 output_dir/logs/tensorboard。"},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "是否丢弃数据集中未被模型直接使用的列。"},
    )

    bf16: bool = field(default=False, metadata={"help": "启用 bf16（推荐 GPU 环境）。"})
    fp16: bool = field(default=False, metadata={"help": "启用 fp16。"})
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "启用或关闭 gradient checkpointing。"},
    )
    use_peft: bool = field(default=True, metadata={"help": "启用或关闭 LoRA / PEFT。"})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank。"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha。"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout。"})
    lora_target_modules: str = field(
        default="q_proj,v_proj",
        metadata={"help": "逗号分隔的 LoRA target modules。"},
    )

    use_vllm: bool = field(default=False, metadata={"help": "使用 vLLM 加速生成（Linux 更合适）。"})
    vllm_mode: str = field(
        default="colocate",
        metadata={"help": "vLLM 运行模式。", "choices": ["colocate", "server"]},
    )
    log_completions: bool = field(default=False, metadata={"help": "是否记录模型生成内容。"})
    push_to_hub: bool = field(default=False, metadata={"help": "训练结束后是否推送到 Hugging Face Hub。"})
    hub_model_id: str | None = field(default=None, metadata={"help": "推送到 Hub 时使用的 repo ID。"})
    device_map_auto: bool = field(
        default=False,
        metadata={"help": "仅适合单卡/非分布式的快速实验。"},
    )


def build_arg_parser() -> HfArgumentParser:
    """返回 Hugging Face dataclass 参数解析器，兼容旧测试里的 parse_args 调用。"""

    return HfArgumentParser(TrainScriptArguments, description="使用 GRPO 对 VLM 做后训练。")


def parse_args(argv: list[str] | None = None) -> TrainScriptArguments:
    """解析命令行参数为 dataclass 对象。"""

    return build_arg_parser().parse_args_into_dataclasses(argv)[0]


def _suppress_http_request_logs() -> None:
    """默认关闭 httpx/httpcore 的请求级 INFO 日志。"""

    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _resolve_dtype(args: TrainScriptArguments) -> Any:
    import torch

    if args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if args.fp16 and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _build_lora_config(args: TrainScriptArguments) -> Any | None:
    if not args.use_peft:
        return None

    from peft import LoraConfig

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )


def _resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: str | None) -> str | None:
    """解析 checkpoint 续训路径。"""

    if resume_from_checkpoint is None:
        return None
    if resume_from_checkpoint.lower() != "last":
        return resume_from_checkpoint

    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        raise ValueError(f"没有在 {output_dir} 下找到 checkpoint-*，无法使用 --resume_from_checkpoint last。")
    latest_checkpoint = max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))
    return str(latest_checkpoint)


def main() -> None:
    args = parse_args()
    _suppress_http_request_logs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = Path(args.logging_dir) if args.logging_dir else output_dir / "logs" / "tensorboard"
    logging_dir.mkdir(parents=True, exist_ok=True)

    logger.info("output_dir: {}", output_dir)
    logger.info("logging_dir: {}", logging_dir)
    logger.info("args: {}", args)

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from trl import GRPOConfig, GRPOTrainer

        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            padding_side="left",
        )

        raw_cfg = DatasetConfig(
            dataset_id=args.dataset_id,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        raw_dataset = load_raw_dataset(raw_cfg)
        train_dataset, eval_dataset = prepare_datasets(
            raw_dataset,
            processor,
            train_size=args.train_size,
            test_size=args.test_size,
            eval_samples=args.eval_samples,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
        )

        dtype = _resolve_dtype(args)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if args.device_map_auto else None,
        )

        peft_config = _build_lora_config(args)
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
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
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
            reward_funcs=[think_format_reward, accuracy_reward],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )

        resume_checkpoint = _resolve_resume_checkpoint(output_dir, args.resume_from_checkpoint)
        if resume_checkpoint:
            logger.info("resume_from_checkpoint: {}", resume_checkpoint)

        logger.info("start training")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        logger.info("training finished, saving model to {}", output_dir)
        trainer.save_model(str(output_dir))

        if args.push_to_hub:
            logger.info("pushing model to hub: {}", args.hub_model_id)
            trainer.push_to_hub()
    except Exception:
        logger.exception("training failed")
        raise


if __name__ == "__main__":
    main()
