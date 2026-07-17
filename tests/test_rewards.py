from qwen_vl_grpo_reasoning.rewards import accuracy_reward, extract_answer_text, think_format_reward


def test_extract_answer_text():
    assert extract_answer_text("<think>abc</think><answer>42</answer>") == "42"


def test_think_format_reward():
    assert think_format_reward(["<think>a</think><answer>b</answer>"]) == [1.0]
    assert think_format_reward(["oops"]) == [0.0]


def test_accuracy_reward_fallback():
    assert accuracy_reward(["<think>a</think><answer> 42 </answer>"], solution=["42"]) == [1.0]
