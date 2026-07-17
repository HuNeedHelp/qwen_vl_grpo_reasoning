"""推理入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .prompts import SYSTEM_PROMPT
from .rewards import extract_answer_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用训练后的 VLM 做推理。")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    # VLM processor 通常期望 RGB 图片；这里提前统一格式。
    image = Image.open(image_path).convert("RGB")

    # processor 负责 chat template、tokenize、图片预处理；模型负责真正生成。
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    # Qwen-VL 的 user content 可以同时包含 image 和 text。
    # apply_chat_template 会把这个 conversation 转成模型熟悉的文本模板。
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
    # 从 conversation 中提取图片，转换成 processor/model 需要的视觉输入。
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    # generated_ids 包含 prompt + completion，这里只截取新生成的 completion。
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
