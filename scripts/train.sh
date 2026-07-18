#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 接近 Hugging Face Cookbook 的完整训练模板。
# 如果不用 vLLM，删除 --use_vllm 和 --vllm_mode 两行即可。
OUTPUT_DIR="outputs/500steps-Qwen2.5-VL-3B-Instruct-Thinking"
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
    --test_size 0.2
    --max_prompt_tokens 2048

    # 训练步数、batch 和生成配置
    --max_steps 500
    --learning_rate 1e-5
    --num_train_epochs 1
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --num_generations 2
    --max_completion_length 1024

    # 训练期评估：指标会写入 TensorBoard
    --eval_strategy "steps"
    --eval_steps 50
    --per_device_eval_batch_size 8

    # 精度与显存优化
    --bf16
    --use_peft

    # vLLM 生成加速；不用 vLLM 时删除这两行
    # --use_vllm
    # --vllm_mode "colocate"

    # 日志、TensorBoard 和 completion 记录
    --report_to "tensorboard"
    --logging_steps 50
    --log_completions

    # checkpoint 保存策略
    --resume_from_checkpoint "last"
    --save_strategy "steps"
    --save_steps 50

    # 随机性控制
    --seed 42
  )

  accelerate launch --config_file accelerate_config.yaml -m train "${TRAIN_ARGS[@]}"

  echo "finished_at: $(date '+%Y-%m-%d %H:%M:%S')"
} > "${LOG_FILE}" 2>&1


# 训练完成后autodl 服务器关机
/usr/bin/shutdown
