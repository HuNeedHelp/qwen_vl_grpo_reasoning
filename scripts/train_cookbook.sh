#!/usr/bin/env bash
set -euo pipefail

# 接近 Hugging Face Cookbook 的完整训练模板。
# 如果不用 vLLM，删除 --use_vllm 和 --vllm_mode 两行即可。
accelerate launch --config_file accelerate_config.yaml -m qwen_vl_grpo_reasoning.train \
  --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --dataset_id "lmms-lab/multimodal-open-r1-8k-verified" \
  --dataset_split "train[:5%]" \
  --output_dir "outputs/Qwen2.5-VL-3B-Instruct-Thinking" \
  --max_prompt_tokens 2048 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --bf16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_completion_length 1024 \
  --num_generations 2 \
  --use_peft \
  --use_vllm \
  --vllm_mode colocate \
  --log_completions \
  --report_to "tensorboard" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 10 \
  --test_size 100 \
  --seed 42
