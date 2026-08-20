    

# Qwen-VL GRPO Reasoning

这是一个基于 Hugging Face Cookbook `fine_tuning_vlm_grpo_trl` 整理出来的中文完整项目。项目目标是用 TRL 的 GRPO 方法，对 `Qwen/Qwen2.5-VL-3B-Instruct` 做多模态推理后训练，让模型在图片问题上更稳定地输出：

```text
<think>推理过程</think><answer>最终答案</answer>
```

项目包含从数据处理、GRPO 训练、LoRA 配置、推理验证、科学评测、断点续评到简历展示的完整流程。

## 项目目标

1. 使用 `Qwen/Qwen2.5-VL-3B-Instruct` 作为 Vision-Language base model。
2. 使用 `lmms-lab/multimodal-open-r1-8k-verified` 的图文推理样本做 GRPO 后训练。
3. 设计两个 reward function：
   - 格式奖励：检查 `<think>...</think><answer>...</answer>` 结构。
   - 答案奖励：比较 `<answer>` 与标准答案。
4. 构建 base model vs GRPO model 的可复现实验评测。
5. 输出可以写进简历的量化指标、置信区间和逐题 case study。

## 项目结构

```text
qwen_vl_grpo_reasoning/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ accelerate_config.yaml
├─ scripts/
│  ├─ train.sh              # 接近 cookbook 的完整训练模板
│  ├─ test_run_training.sh  # 小规模 smoke test 训练
│  ├─ infer.sh              # 单张图片推理
│  ├─ eval.sh               # base vs GRPO 科学评测
│  └─ test.sh               # 单元测试
├─ src/
│  ├─ train.py              # GRPO 训练入口
│  ├─ infer.py              # 本地推理入口
│  ├─ evaluate.py           # 批量评测、断点续评、指标汇总
│  └─ utilities/
│     ├─ data.py            # 数据加载、过滤、chat prompt 构造
│     ├─ prompts.py         # 训练、推理共用 system prompt
│     ├─ rewards.py         # GRPO reward functions
│     ├─ vlm.py             # VLM 加载和图文生成
│     ├─ eval_metrics.py    # 评测指标与 paired comparison
│     └─ eval_io.py         # JSONL、summary、断点续评文件管理
└─ tests/
   ├─ conftest.py
   ├─ test_rewards.py
   └─ test_evaluate.py
```

## 环境准备

推荐环境：

1. Python 3.10+
2. NVIDIA GPU
3. Linux / WSL2 / 远程 GPU 服务器

Windows 可以阅读、改代码和跑部分测试，但正式训练 VLM + GRPO 更建议 Linux 或 WSL2。尤其是 `qwen-vl-utils`、`torchvision`、`vLLM` 这类依赖在 Linux 环境下更稳。

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` 很重要。项目采用 `src layout`，安装后 Python 才能通过下面的模块路径找到训练入口：

```bash
python3 -m train
```

注意这里不是：

```bash
python3 -m src.train
```

`src/` 是源码目录，不是 Python 包名。项目脚本也会自动把 `src/` 加入 `PYTHONPATH`，因此即使没有重新安装 editable package，也可以直接运行脚本。

源码现在按“入口 + 工具模块”拆分：

```text
src/train.py       # 训练流程编排
src/infer.py       # 单图推理流程编排
src/evaluate.py    # 评测流程编排
src/utilities/     # 可复用能力
```

这样入口文件不再塞满细节逻辑，数据处理、奖励函数、VLM 生成、评测指标和文件读写都各自独立。

入口脚本的命令行参数使用 Hugging Face `HfArgumentParser` 从 `@dataclass` 自动生成：

```text
src/train.py     TrainScriptArguments
src/infer.py     InferScriptArguments
src/evaluate.py  EvalScriptArguments
```

这样参数默认值、类型和帮助文档都集中写在 dataclass 字段里，不再手动堆 `parser.add_argument(...)`。脚本里的启动命令仍然保持显式参数，想改实验配置时直接改 `scripts/*.sh` 即可。

如果你使用 Git Bash、WSL 或不同 conda 环境，脚本默认调用 `python3`。如果你的解释器叫 `python`，可以这样运行：

```bash
PYTHON_BIN=python bash scripts/test.sh
```

如果需要下载受限模型或推送 Hub，需要先登录 Hugging Face：

```bash
huggingface-cli login
```

## 一键流程

推荐按这个顺序跑完整项目：

```bash
# 1. 安装项目
pip install -r requirements.txt
pip install -e .

# 2. 跑单元测试
bash scripts/test.sh

# 3. 小规模训练试跑
bash scripts/test_run_training.sh

# 4. 修改 scripts/infer.sh 中的 IMAGE_PATH 后做推理
bash scripts/infer.sh

# 5. 训练完成后做 base vs GRPO 评测
bash scripts/eval.sh
```

如果你准备正式复现 cookbook 风格训练，可以用：

```bash
bash scripts/train.sh
```

## 整体流程

项目的核心数据流如下：

```text
原始数据集
  └─ data.py
      ├─ 过滤过大图片
      ├─ 转 RGB
      ├─ train/eval split
      ├─ 构造 system + user prompt
      └─ 过滤超长 prompt
          ↓
GRPOTrainer
  ├─ Qwen2.5-VL base model
  ├─ AutoProcessor
  ├─ LoRA / PEFT
  └─ reward functions
      ├─ think_format_reward
      └─ accuracy_reward
          ↓
训练 checkpoint / LoRA adapter
          ↓
infer.py 单样本验证
          ↓
evaluate.py 批量评测
  ├─ base predictions
  ├─ GRPO predictions
  ├─ paired comparison
  ├─ bootstrap confidence interval
  └─ summary.csv / summary.json
```

## 数据处理流程

数据处理入口在 `src/utilities/data.py`。

### 1. 加载数据

默认数据集：

```text
lmms-lab/multimodal-open-r1-8k-verified
```

每条样本主要使用三个字段：

```text
problem   # 文本问题
image     # PIL 图片
solution  # 标准答案
```

当前数据集在 Hub 上主要提供 `train` split，因此项目不再暴露 `dataset_split` 参数，也不使用 `train[:5%]` 这类远端 slice。代码会固定加载完整 `train` split，然后在本地用固定 `seed` 做确定性的 train/eval 切分。

### 2. 本地 train/eval 切分

代码会在 `src/utilities/data.py` 里按这个顺序处理：

```text
load_dataset(dataset_id, split="train")
  -> 过滤过大图片
  -> 转 RGB
  -> train_test_split(train_size=train_size, test_size=test_size, seed=seed)
  -> train/eval 分别转换成 GRPO 样本
```

其中 `test_size` 控制留出的验证/评测样本数；`train_size` 可选，控制训练样本数。`train_size=None` 时，训练集会使用除 test set 之外的全部样本。小规模 smoke test 不再靠 `train[:1%]`，而是用 `--train_size 80 --test_size 20` 明确控制本地切分大小。

为了避免训练集和评测集重叠，正式训练和评测应保持这几个参数一致：

```text
dataset_id
train_size
test_size
seed
max_prompt_tokens
```

例如训练使用 `--train_size 500 --test_size 100 --seed 42`，评测也应该使用同样的 `train_size/test_size/seed`，这样复现出来的 eval set 才是同一批留出样本。

### 3. 图片过滤与转换

代码会过滤宽高大于等于 `512` 的图片：

```python
image.size[0] < 512 and image.size[1] < 512
```

这样做是为了降低 VLM 训练时的视觉 token 数量和显存压力。随后所有图片都会转换成 RGB。

### 4. 构造 prompt

原始 `problem` 会被转换成对话格式：

```python
[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": problem},
]
```

训练时 `GRPOTrainer` 会结合 processor 和数据中的 `image` 字段处理图文输入。

### 5. 过滤超长 prompt

TRL 里的 `max_prompt_length` 已经 deprecated。本项目使用项目级参数：

```bash
--max_prompt_tokens 2048
```

它会在数据进入 trainer 之前调用：

```python
processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
```

然后过滤超过 token 数限制的样本。

## Prompt 设计

公共提示词在 `src/utilities/prompts.py`：

```text
The assistant first thinks about the reasoning process...
<think> reasoning process here </think><answer> answer here </answer>
```

这个 prompt 的目的不是直接告诉模型答案，而是约束输出结构，方便 reward function 自动解析推理过程和最终答案。

## Reward 设计

奖励函数在 `src/utilities/rewards.py`。

### 格式奖励

`think_format_reward` 检查输出是否严格匹配：

```text
<think>...</think><answer>...</answer>
```

匹配得 `1.0`，不匹配得 `0.0`。

### 答案奖励

`accuracy_reward` 会先从 `<answer>` 标签中提取最终答案，然后比较标准答案：

1. 如果环境安装了 `math_verify` 和 `latex2sympy2_extended`，优先做数学表达式校验。
2. 如果数学校验不可用或解析失败，就退回到规范化字符串比较。

总训练 reward 来自两个 reward function 的组合，GRPO 会鼓励模型同时学会结构化输出和答对题目。

## 训练流程

训练入口在 `src/train.py`。

核心步骤：

1. 用 `HfArgumentParser` 将命令行参数解析为 `TrainScriptArguments`。
2. 创建输出目录。
3. 加载 `AutoProcessor`。
4. 加载并处理数据集。
5. 加载 `Qwen2_5_VLForConditionalGeneration`。
6. 构造 LoRA 配置。
7. 构造 `GRPOConfig`。
8. 创建 `GRPOTrainer`。
9. 执行 `trainer.train()`。
10. 保存模型或 adapter。

### 小规模试跑

先用 `scripts/test_run_training.sh` 跑通完整链路：

```bash
bash scripts/test_run_training.sh
```

默认参数：

```bash
--train_size 80
--test_size 20
--max_steps 20
--per_device_train_batch_size 2
--gradient_accumulation_steps 4
--num_generations 4
--max_completion_length 512
--eval_strategy "steps"
--eval_steps 10
--per_device_eval_batch_size 1
--use_peft
--bf16
```

这个脚本主要用于确认：

1. 数据能下载和处理。
2. processor 能正常工作。
3. reward function 能被 trainer 调用。
4. 模型和 LoRA 能正常保存。

### Cookbook 风格训练

更完整的训练模板在：

```bash
bash scripts/train.sh
```

核心参数：

```bash
accelerate launch --config_file accelerate_config.yaml -m train \
  --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --dataset_id "lmms-lab/multimodal-open-r1-8k-verified" \
  --output_dir "outputs/Qwen2.5-VL-3B-Instruct-Thinking" \
  --max_prompt_tokens 2048 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --bf16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --max_completion_length 1024 \
  --num_generations 2 \
  --eval_strategy "steps" \
  --eval_steps 50 \
  --per_device_eval_batch_size 1 \
  --use_peft \
  --use_vllm \
  --vllm_mode colocate \
  --log_completions \
  --report_to "tensorboard" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 50 \
  --test_size 100 \
  --seed 42
```

如果你的环境不支持 vLLM，可以删除：

```bash
--use_vllm
--vllm_mode colocate
```

### LoRA 配置

默认使用 PEFT / LoRA：

```bash
--use_peft
--lora_r 8
--lora_alpha 32
--lora_dropout 0.1
--lora_target_modules "q_proj,v_proj"
```

这会显著降低训练显存需求。训练结果通常保存为 adapter，因此评测脚本支持自动检测 `adapter_config.json` 并加载 base model + adapter。

### 训练输出

训练输出默认放在：

```text
outputs/Qwen2.5-VL-3B-Instruct-Thinking/
```

常见内容包括：

```text
adapter_config.json
adapter_model.safetensors
checkpoint-*/
trainer_state.json
logs/
  ├─ train.log
  └─ tensorboard/
```

## 训练监控

训练脚本默认：

```bash
--report_to "tensorboard"
--eval_strategy "steps"
--eval_steps 50
```

`eval_strategy` 控制训练中是否定期在 `eval_dataset` 上跑评估：

```text
no     不做训练期评估
steps  每隔 eval_steps 个 step 评估一次
epoch  每个 epoch 结束后评估一次
```

本项目默认在训练中使用 `steps`。完整训练模板每 50 step 评估一次，小规模 smoke test 每 10 step 评估一次。评估会触发额外的生成和 reward 计算，因此显存或时间压力较大时，可以把 `eval_steps` 调大，或者临时改成 `--eval_strategy "no"`。

普通文本日志默认保存到：

```text
outputs/Qwen2.5-VL-3B-Instruct-Thinking/logs/train.log
```

普通文本日志由 `scripts/train.sh` 或 `scripts/test_run_training.sh` 使用 shell 重定向生成：

```bash
} > "${LOG_FILE}" 2>&1
```

因此 `train.log` 会记录 Trainer/tqdm 进度条、warning、stdout/stderr，以及代码里用 `loguru` 输出的关键训练节点。Python 端不再维护复杂的 file handler 或 Tee 逻辑。

代码默认会把 `httpx` 和 `httpcore` 的 HTTP request INFO 日志压到 WARNING，避免 Hugging Face 下载模型和数据集时把日志刷屏。

如果想看实时终端输出并同时保存日志，可以把脚本末尾的重定向改成：

```bash
} 2>&1 | tee "${LOG_FILE}"
```

TensorBoard 日志默认保存到：

```text
outputs/Qwen2.5-VL-3B-Instruct-Thinking/logs/tensorboard/
```

可以用 TensorBoard 查看 loss、reward、学习率等曲线：

```bash
tensorboard --logdir outputs/Qwen2.5-VL-3B-Instruct-Thinking/logs/tensorboard
```

浏览器打开 TensorBoard 输出的本地地址即可。

如果想手动指定 TensorBoard 日志位置，可以在训练命令中加入：

```bash
--logging_dir "outputs/your-exp/logs/tensorboard"
```

如果训练被中断，可以从指定 checkpoint 续训：

```bash
--resume_from_checkpoint "outputs/your-exp/checkpoint-100"
```

也可以自动使用 `output_dir` 下编号最大的 checkpoint：

```bash
--resume_from_checkpoint last
```

建议每组实验使用不同的 `output_dir`，这样 checkpoint、adapter、文本日志和 TensorBoard 曲线都会自然隔离。

建议重点观察：

1. `reward` 是否整体上升。
2. `loss` 是否出现异常爆炸。
3. `completion_length` 是否异常变长。
4. 如果开启 `--log_completions`，抽查模型输出是否真的符合 `<think>/<answer>` 格式。

## 推理流程

推理入口在 `src/infer.py`。

运行前编辑 `scripts/infer.sh`：

```bash
MODEL_PATH="outputs/Qwen2.5-VL-3B-Instruct-Thinking"
IMAGE_PATH="path/to/sample.png"
PROMPT="请根据图片中的问题进行推理，并给出最终答案。"
```

然后运行：

```bash
bash scripts/infer.sh
```

推理流程：

1. 加载图片并转 RGB。
2. 加载 processor 和模型。
3. 构造 system + image + text conversation。
4. 调用 `processor.apply_chat_template(...)`。
5. 用 `qwen_vl_utils.process_vision_info(...)` 处理图片。
6. 调用 `model.generate(...)`。
7. 解码 completion。
8. 打印完整输出和提取出的 `<answer>`。

## 评测流程

评测入口在 `src/evaluate.py`。

运行前编辑 `scripts/eval.sh`：

```bash
BASE_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
GRPO_MODEL_PATH="outputs/Qwen2.5-VL-3B-Instruct-Thinking"
OUTPUT_DIR="outputs/eval/base_vs_grpo"
```

然后运行：

```bash
bash scripts/eval.sh
```

默认评测参数：

```bash
--test_size 100
--eval_samples 100
--max_prompt_tokens 2048
--max_new_tokens 512
--eval_batch_size 2
--temperature 0.0
--top_p 1.0
--bootstrap_samples 1000
--seed 42
--resume
```

### 评测内部流程

1. 用 `HfArgumentParser` 将命令行参数解析为 `EvalScriptArguments`。
2. 解析 `base=path`、`grpo=path` 形式的模型列表。
3. 创建输出目录，并立即写入或校验 `run_config.json`。
4. 加载 processor。
5. 按和训练一致的数据逻辑构造 eval set。
6. 对每个模型逐批生成回答。
7. 每个 batch 完成后立即追加写入 `*_predictions.jsonl`。
8. 对每条 completion 计算格式分、准确率和总 reward。
9. 对每个模型汇总整体指标。
10. 对 GRPO 和 base 做同题 paired comparison。
11. 保存 JSONL、CSV 和 JSON 汇总。

### 批量推理

`--eval_batch_size` 控制一次送入模型的图文样本数：

```bash
--eval_batch_size 2
```

调大 batch size 通常可以提升 GPU 吞吐，但会增加显存占用。VLM 的显存压力来自：

1. 图片视觉 token。
2. 文本 padding。
3. generation KV cache。
4. `max_new_tokens`。

如果出现 CUDA OOM，优先改成：

```bash
--eval_batch_size 1
```

评测结果里的 `latency_seconds` 是 batch 总耗时除以 batch 样本数得到的平均单样本耗时。比较延迟时，应保持 batch size、硬件和生成参数一致。

### 断点续评

评测默认开启：

```bash
--resume
```

每完成一个 batch，脚本都会把新结果追加写入：

```text
base_predictions.jsonl
grpo_predictions.jsonl
```

如果任务中断，再次运行相同命令时，脚本会：

1. 读取已有预测。
2. 按 `sample_id` 去重。
3. 跳过已经完成的样本。
4. 只生成剩余样本。
5. 完成后重新整理 JSONL。

`run_config.json` 会在 processor、数据集和模型加载之前创建，记录模型路径、数据集、seed、batch size 和生成参数。因此，即使首次运行在下载数据或加载模型阶段中断，输出目录里也应该已经存在该文件。

续评时如果配置改变，脚本会拒绝混用旧结果。此时可以换一个 `OUTPUT_DIR`，或者强制重跑：

```bash
--no_resume
```

也支持：

```bash
--no-resume
```

### 评测输出

默认输出目录：

```text
outputs/eval/base_vs_grpo/
├─ run_config.json
├─ base_predictions.jsonl
├─ grpo_predictions.jsonl
├─ grpo_vs_base_paired.jsonl
├─ summary.json
└─ summary.csv
```

文件说明：

```text
run_config.json              # 本次评测配置，用于断点续评校验
base_predictions.jsonl       # base model 每题输出和分数
grpo_predictions.jsonl       # GRPO model 每题输出和分数
grpo_vs_base_paired.jsonl    # 同题对比明细
summary.json                 # 完整指标，包含置信区间和 paired comparison
summary.csv                  # 扁平表格，适合粘到实验记录
```

## 评测指标

`format_rate`：输出是否满足 `<think>...</think><answer>...</answer>` 格式。这个指标衡量 GRPO 是否学会了稳定的可解析输出。

`accuracy`：从 `<answer>` 中提取最终答案，并和 `solution` 比较。若数学验证依赖可用，会优先做数学表达式校验；否则退回字符串比较。

`avg_total_reward`：`format_score + accuracy` 的平均值，与训练 reward 保持一致。

`invalid_output_rate`：`1 - format_rate`，衡量无法自动解析的输出比例。

`avg_completion_tokens`：平均生成长度，用来判断模型是否通过生成过长推理过程来换取 reward。

`avg_latency_seconds`：平均单样本生成耗时，用来衡量推理成本。

`paired win/tie/loss`：在同一道题上比较 GRPO model 和 base model 的 `total_reward`，统计 GRPO 赢、平、输的比例。

`bootstrap 95% CI`：对核心均值和配对 delta 做 bootstrap 置信区间，避免只报告一个偶然点估计。

## 为什么这个评测更可信

1. base 和 GRPO 使用同一份留出集。
2. 使用固定 `seed`，数据切分可复现。
3. 使用 `temperature=0.0`，减少采样随机性。
4. 固定 `max_prompt_tokens` 和 `max_new_tokens`，控制输入输出长度。
5. 不只看 accuracy，还看格式合规率、总 reward、输出长度和延迟。
6. 使用 bootstrap 置信区间表达不确定性。
7. 使用 paired comparison 判断同题上的稳定改进。
8. 保存逐样本 JSONL，支持人工 case study。
9. 使用 `run_config.json` 防止续评时混入不同实验配置。

## 结果

`summary.csv`：

```text
model,num_samples,format_rate,accuracy,avg_total_reward,invalid_output_rate
base,...
grpo,...
```

在实验记录中整理成：

```text
Format Compliance:  base_xx% -> grpo_yy%
Answer Accuracy:    base_xx% -> grpo_yy%
Avg Reward:         base_x.xx -> grpo_y.yy
Invalid Outputs:    base_xx% -> grpo_yy%
Paired Win Rate:    grpo_zz%
```

如果 `format_rate` 明显提升，但 `accuracy` 提升不明显，也不一定是失败。这说明当前 GRPO 主要学到了结构化输出，下一步可以：

1. 增大训练样本。
2. 延长训练步数。
3. 优化 `accuracy_reward`。
4. 改进 prompt。
5. 使用更稳定的数学答案验证依赖。


## 测试

运行：

```bash
bash scripts/test.sh
```

测试覆盖：

1. reward function。
2. answer 提取。
3. bootstrap CI。
4. paired comparison。
5. JSONL 读写。
6. 断点续评配置校验。
7. CLI 参数别名。

## 参考

- Cookbook: [Post training a VLM for reasoning with GRPO using TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_grpo_trl)
- TRL GRPO 文档: [GRPO Trainer](https://huggingface.co/docs/trl/en/grpo_trainer)
- Qwen2.5-VL: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
