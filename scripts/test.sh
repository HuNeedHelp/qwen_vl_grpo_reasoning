#!/usr/bin/env bash
set -euo pipefail

# 项目快速测试：验证奖励函数、评测指标和基础导入是否正常。
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON_BIN}" -m pytest tests
