from utilities.rewards import accuracy_reward, extract_answer_text, think_format_reward


def test_extract_answer_text():
    assert extract_answer_text("<think>abc</think><answer>42</answer>") == "42"


def test_think_format_reward():
    assert think_format_reward(["<think>a</think><answer>b</answer>"]) == [1.0]
    assert think_format_reward(["oops"]) == [0.0]


def test_accuracy_reward_fallback():
    assert accuracy_reward(["<think>a</think><answer> 42 </answer>"], solution=["42"]) == [1.0]


def test_accuracy_reward_extracts_target_answer_tag():
    completion = "<think>reason</think><answer>D</answer>"
    solution = "<think>gold reason</think><answer>D</answer>"

    assert accuracy_reward([completion], solution=[solution]) == [1.0]


def test_accuracy_reward_matches_choice_letter_with_label_text():
    completion = "<think>reason</think><answer>D. Right</answer>"
    solution = "<think>gold reason</think><answer>D</answer>"

    assert accuracy_reward([completion], solution=[solution]) == [1.0]


def test_accuracy_reward_matches_choice_letter_in_sentence():
    completion = "<think>reason</think><answer>The correct answer is B. 12 sq km</answer>"
    solution = "<think>gold reason</think><answer>B</answer>"

    assert accuracy_reward([completion], solution=[solution]) == [1.0]


def test_accuracy_reward_rejects_wrong_choice_letter():
    completion = "<think>reason</think><answer>C. Scalene</answer>"
    solution = "<think>gold reason</think><answer>D</answer>"

    assert accuracy_reward([completion], solution=[solution]) == [0.0]
