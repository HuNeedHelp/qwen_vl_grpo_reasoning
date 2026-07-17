from qwen_vl_grpo_reasoning.evaluate import (
    batched,
    build_arg_parser,
    build_paired_rows,
    build_prediction_row,
    build_run_config,
    bootstrap_ci,
    clear_eval_outputs,
    deduplicate_rows,
    paired_compare,
    parse_model_specs,
    read_jsonl,
    score_completion,
    validate_or_write_run_config,
    write_jsonl,
)


def test_parse_model_specs():
    specs = parse_model_specs(["base=Qwen/Qwen2.5-VL-3B-Instruct", "grpo=outputs/model"])

    assert specs[0].label == "base"
    assert specs[0].path == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert specs[1].label == "grpo"
    assert specs[1].path == "outputs/model"


def test_resume_cli_aliases():
    parser = build_arg_parser()

    assert parser.parse_args(["--models", "base=model"]).resume is True
    assert parser.parse_args(["--models", "base=model", "--no_resume"]).resume is False
    assert parser.parse_args(["--models", "base=model", "--no-resume"]).resume is False


def test_bootstrap_ci_is_deterministic():
    values = [0.0, 1.0, 1.0, 0.0]

    first = bootstrap_ci(values, samples=100, seed=42)
    second = bootstrap_ci(values, samples=100, seed=42)

    assert first == second
    assert first["mean"] == 0.5
    assert 0.0 <= first["ci_low"] <= first["ci_high"] <= 1.0


def test_score_completion():
    scores = score_completion("<think>reason</think><answer>42</answer>", "42")

    assert scores["format_score"] == 1.0
    assert scores["accuracy"] == 1.0
    assert scores["total_reward"] == 2.0
    assert scores["answer"] == "42"


def test_paired_compare_and_rows():
    baseline_rows = [
        {
            "sample_id": 0,
            "question": "q0",
            "solution": "a",
            "completion": "base0",
            "total_reward": 1.0,
            "accuracy": 0.0,
            "format_score": 1.0,
        },
        {
            "sample_id": 1,
            "question": "q1",
            "solution": "b",
            "completion": "base1",
            "total_reward": 2.0,
            "accuracy": 1.0,
            "format_score": 1.0,
        },
    ]
    candidate_rows = [
        {
            "sample_id": 0,
            "question": "q0",
            "solution": "a",
            "completion": "cand0",
            "total_reward": 2.0,
            "accuracy": 1.0,
            "format_score": 1.0,
        },
        {
            "sample_id": 1,
            "question": "q1",
            "solution": "b",
            "completion": "cand1",
            "total_reward": 1.0,
            "accuracy": 0.0,
            "format_score": 1.0,
        },
    ]

    summary = paired_compare(
        baseline_rows,
        candidate_rows,
        bootstrap_samples=100,
        seed=42,
    )
    paired_rows = build_paired_rows("base", baseline_rows, "grpo", candidate_rows)

    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 0
    assert summary["win_rate"] == 0.5
    assert paired_rows[0]["outcome"] == "win"
    assert paired_rows[1]["outcome"] == "loss"


def test_batched():
    assert batched([0, 1, 2, 3, 4], 2) == [[0, 1], [2, 3], [4]]


def test_deduplicate_rows_keeps_last_valid_row():
    rows = [
        {"sample_id": 0, "completion": "old"},
        {"sample_id": 1, "completion": "keep"},
        {"sample_id": 0, "completion": "new"},
        {"sample_id": 99, "completion": "ignore"},
        {"sample_id": "bad", "completion": "ignore"},
    ]

    deduplicated = deduplicate_rows(rows, max_sample_id=2)

    assert deduplicated == [
        {"sample_id": 0, "completion": "new"},
        {"sample_id": 1, "completion": "keep"},
    ]


def test_write_and_read_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [{"sample_id": 0, "completion": "a"}, {"sample_id": 1, "completion": "b"}]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_read_jsonl_ignores_truncated_last_line(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"sample_id": 0, "completion": "ok"}\n{"sample_id": ', encoding="utf-8")

    assert read_jsonl(path) == [{"sample_id": 0, "completion": "ok"}]


def test_validate_or_write_run_config(tmp_path):
    path = tmp_path / "run_config.json"
    config = {"dataset_id": "demo", "seed": 42}

    validate_or_write_run_config(path, config, resume=True, has_predictions=False)
    validate_or_write_run_config(path, config, resume=True, has_predictions=True)

    try:
        validate_or_write_run_config(path, {"dataset_id": "demo", "seed": 7}, resume=True, has_predictions=True)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("配置变化时必须阻止续评。")


def test_build_run_config_uses_requested_eval_samples():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--models",
            "base=model-a",
            "grpo=model-b",
            "--processor_path",
            "processor",
            "--eval_samples",
            "123",
            "--eval_batch_size",
            "4",
        ]
    )
    specs = parse_model_specs(args.models)

    config = build_run_config(specs, args.processor_path, args)

    assert config["eval_samples"] == 123
    assert config["eval_batch_size"] == 4
    assert config["models"] == [
        {"label": "base", "path": "model-a"},
        {"label": "grpo", "path": "model-b"},
    ]


def test_clear_eval_outputs(tmp_path):
    specs = parse_model_specs(["base=model-a", "grpo=model-b"])
    generated_files = [
        "run_config.json",
        "summary.json",
        "summary.csv",
        "base_predictions.jsonl",
        "grpo_predictions.jsonl",
        "grpo_vs_base_paired.jsonl",
    ]
    for filename in generated_files:
        (tmp_path / filename).write_text("old", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")

    clear_eval_outputs(tmp_path, specs)

    assert not any((tmp_path / filename).exists() for filename in generated_files)
    assert (tmp_path / "keep.txt").exists()


def test_build_prediction_row():
    example = {
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        "solution": "42",
    }

    row = build_prediction_row(
        sample_id=3,
        model_label="base",
        example=example,
        completion="<think>x</think><answer>42</answer>",
        completion_tokens=12,
        latency=0.5,
    )

    assert row["sample_id"] == 3
    assert row["model"] == "base"
    assert row["question"] == "q"
    assert row["accuracy"] == 1.0
    assert row["completion_tokens"] == 12
