#!/usr/bin/env bash
set -euo pipefail

# 小规模试跑配置：先确认数据、奖励函数、模型加载和训练循环能跑通。
# 需要改参数时，直接改下面的命令即可。
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="outputs/_check_the_code"

"${PYTHON_BIN}" -m qwen_vl_grpo_reasoning.train \
  --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --dataset_id "lmms-lab/multimodal-open-r1-8k-verified" \
  --dataset_split "train[:1%]" \
  --output_dir "${OUTPUT_DIR}" \
  --max_prompt_tokens 2048 \
  --learning_rate 1e-5 \
  --max_steps 20 \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_completion_length 512 \
  --num_generations 2 \
  --use_peft \
  --report_to "tensorboard" \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps 20 \
  --test_size 20 \
  --seed 42
