from datasets import Dataset
from PIL import Image

from utilities.data import split_raw_dataset


def _build_raw_dataset(size: int = 12) -> Dataset:
    return Dataset.from_list(
        [
            {
                "source_id": idx,
                "problem": f"question {idx}",
                "solution": str(idx),
                "image": Image.new("RGB", (16, 16), color=(idx, 0, 0)),
            }
            for idx in range(size)
        ]
    )


def test_split_raw_dataset_is_disjoint_and_deterministic():
    first = split_raw_dataset(_build_raw_dataset(), train_size=5, test_size=3, seed=42)
    second = split_raw_dataset(_build_raw_dataset(), train_size=5, test_size=3, seed=42)

    train_ids = set(first["train"]["source_id"])
    eval_ids = set(first["test"]["source_id"])

    assert len(train_ids) == 5
    assert len(eval_ids) == 3
    assert train_ids.isdisjoint(eval_ids)
    assert first["train"]["source_id"] == second["train"]["source_id"]
    assert first["test"]["source_id"] == second["test"]["source_id"]
