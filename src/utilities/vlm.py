"""VLM 模型加载与图文生成工具。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def load_model(model_path: str, *, device_map: str = "auto") -> Any:
    """加载普通模型或 PEFT/LoRA adapter。"""

    from peft import PeftConfig, PeftModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_dir = Path(model_path)
    adapter_config = model_dir / "adapter_config.json"

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
    """把 text-only user prompt 转成 Qwen-VL 推理需要的图文 prompt。"""

    conversation: list[dict[str, Any]] = []
    for message in prompt:
        if message["role"] != "user":
            conversation.append(message)
            continue

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


def generate_batch(
    model: Any,
    processor: Any,
    examples: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, int, float]]:
    """批量生成 completion，返回文本、生成 token 数和平均单样本耗时。"""

    import torch
    from qwen_vl_utils import process_vision_info

    conversations = [add_image_to_prompt(example["prompt"], example["image"]) for example in examples]
    prompt_texts = [
        processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in conversations
    ]
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
    """单样本生成接口，内部复用 batch 实现。"""

    return generate_batch(
        model,
        processor,
        [{"prompt": prompt, "image": image}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0]

