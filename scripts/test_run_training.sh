#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 小规模试跑配置：先确认数据、奖励函数、模型加载和训练循环能跑通。
# 需要改参数时，直接改下面的命令即可。
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="outputs/test_run_training"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"

{
  echo "log_file: ${LOG_FILE}"
  echo "started_at: $(date '+%Y-%m-%d %H:%M:%S')"

  TRAIN_ARGS=(
    # 模型与输出目录
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct"
    --output_dir "${OUTPUT_DIR}"

    # 数据集与 prompt 过滤
    --dataset_id "lmms-lab/multimodal-open-r1-8k-verified"
    --dataset_split "train[:1%]"
    --test_size 20
    --max_prompt_tokens 2048

    # 小规模训练步数、batch 和生成配置
    --learning_rate 1e-5
    --max_steps 20
    --per_device_train_batch_size 2
    --gradient_accumulation_steps 4
    --num_generations 4
    --max_completion_length 512

    # 训练期评估：smoke test 中更频繁，方便确认 TensorBoard 有 eval 指标
    --eval_strategy "steps"
    --eval_steps 10
    --per_device_eval_batch_size 8

    # 精度与显存优化
    --bf16
    --use_peft

    # 日志和 TensorBoard
    --report_to "tensorboard"
    --logging_steps 5

    # checkpoint 保存策略
    --save_strategy "steps"
    --save_steps 20

    # 随机性控制
    --seed 42
  )

  "${PYTHON_BIN}" -m train "${TRAIN_ARGS[@]}"

  echo "finished_at: $(date '+%Y-%m-%d %H:%M:%S')"
} > "${LOG_FILE}" 2>&1
