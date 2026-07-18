#!/usr/bin/env bash
set -euo pipefail

# 无论从哪个目录调用脚本，都把相对路径解析到项目根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 科学评测模板：用同一批留出样本比较 base model 和 GRPO 后训练模型。
# 训练完成后，把 GRPO_MODEL_PATH 改成你的 checkpoint/adaptor 目录。
BASE_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
GRPO_MODEL_PATH="outputs/500steps-Qwen2.5-VL-3B-Instruct-Thinking"
OUTPUT_DIR="outputs/eval/500steps_base_vs_grpo"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m evaluate \
  --models "base=${BASE_MODEL_PATH}" "grpo=${GRPO_MODEL_PATH}" \
  --processor_path "${BASE_MODEL_PATH}" \
  --dataset_id "lmms-lab/multimodal-open-r1-8k-verified" \
  --dataset_split "train[:5%]" \
  --test_size 100 \
  --eval_samples 100 \
  --max_prompt_tokens 2048 \
  --max_new_tokens 512 \
  --eval_batch_size 2 \
  --temperature 0.0 \
  --top_p 1.0 \
  --bootstrap_samples 1000 \
  --seed 42 \
  --device_map "auto" \
  --no_resume \
  --output_dir "${OUTPUT_DIR}"


/usr/bin/shutdown