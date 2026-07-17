"""训练与推理共用的提示词。"""

# 这个 system prompt 是 reward 设计的一部分：
# 模型只有按 <think>...</think><answer>...</answer> 输出，
# 格式奖励和答案提取逻辑才能稳定工作。
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)
