#!/usr/bin/env bash
set -euo pipefail

# 推理模板：训练完成后，把 MODEL_PATH 和 IMAGE_PATH 改成真实路径再运行。
MODEL_PATH="outputs/Qwen2.5-VL-3B-Instruct-Thinking"
IMAGE_PATH="path/to/sample.png"
PROMPT="请根据图片中的问题进行推理，并给出最终答案。"
MAX_NEW_TOKENS=512
TEMPERATURE=0.2
TOP_P=0.9
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "请先在 scripts/infer.sh 中把 IMAGE_PATH 改成真实图片路径。"
  exit 1
fi

"${PYTHON_BIN}" -m qwen_vl_grpo_reasoning.infer \
  --model_path "${MODEL_PATH}" \
  --image_path "${IMAGE_PATH}" \
  --prompt "${PROMPT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}"
