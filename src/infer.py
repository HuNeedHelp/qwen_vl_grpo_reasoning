"""推理入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from transformers import HfArgumentParser

from utilities.prompts import SYSTEM_PROMPT
from utilities.rewards import extract_answer_text


@dataclass
class InferScriptArguments:
    """单样本推理脚本参数。"""

    model_path: str = field(metadata={"help": "训练后模型目录或 Hugging Face Hub ID。"})
    image_path: str = field(metadata={"help": "待推理图片路径。"})
    prompt: str = field(metadata={"help": "用户问题。"})
    max_new_tokens: int = field(default=512, metadata={"help": "最多生成的新 token 数。"})
    temperature: float = field(default=0.2, metadata={"help": "生成 temperature；0 表示贪心生成。"})
    top_p: float = field(default=0.9, metadata={"help": "采样 top_p。"})


def build_arg_parser() -> HfArgumentParser:
    """返回 Hugging Face dataclass 参数解析器，兼容 parse_args 调用。"""

    return HfArgumentParser(InferScriptArguments, description="使用训练后的 VLM 做推理。")


def parse_args(argv: list[str] | None = None) -> InferScriptArguments:
    """解析命令行参数为 dataclass 对象。"""

    return build_arg_parser().parse_args_into_dataclasses(argv)[0]


def main() -> None:
    args = parse_args()

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(generated_text)
    print()
    print("提取到的 answer：")
    print(extract_answer_text(generated_text))


if __name__ == "__main__":
    main()
