from pathlib import Path

import pandas as pd

from ruraltail_mlr.data.label_schema import DEFAULT_CLASS_NAMES, default_class_mapping, write_class_mapping
from ruraltail_mlr.data.frequency import compute_class_frequency
from ruraltail_mlr.data.split import make_fixed_split, make_split_from_files


def _labels(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {"image_id": f"image_{i:05d}.png"}
        for cls in DEFAULT_CLASS_NAMES:
            row[cls] = 0
        row[DEFAULT_CLASS_NAMES[i % 6]] = 1
        if i % 5 == 0:
            row[DEFAULT_CLASS_NAMES[-1]] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_iterative_split_is_deterministic_and_writes_meta(tmp_path: Path):
    labels = _labels()
    split_a = make_fixed_split(labels, DEFAULT_CLASS_NAMES, seed=7)
    split_b = make_fixed_split(labels, DEFAULT_CLASS_NAMES, seed=7)
    assert split_a.equals(split_b)
    assert set(split_a["split"]) == {"train", "val", "test"}

    labels_csv = tmp_path / "labels.csv"
    mapping_json = tmp_path / "class_mapping.json"
    out_csv = tmp_path / "fixed_split.csv"
    labels.to_csv(labels_csv, index=False)
    write_class_mapping(default_class_mapping(), mapping_json)
    make_split_from_files(labels_csv, mapping_json, out_csv, seed=7)
    assert (tmp_path / "split_meta.json").exists()
    assert (tmp_path / "split_distribution.csv").exists()


def test_frequency_groups_use_train_counts_equal_thirds():
    class_names = [f"cls_{idx}" for idx in range(9)]
    rows = []
    for image_idx in range(20):
        row = {"image_id": f"image_{image_idx:05d}.png"}
        for class_idx, cls in enumerate(class_names):
            row[cls] = int(image_idx < 9 - class_idx)
        rows.append(row)
    labels = pd.DataFrame(rows)
    split = pd.DataFrame(
        {
            "image_id": labels["image_id"],
            "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
        }
    )

    freq, hmt = compute_class_frequency(labels, split, class_names)
    assert [len(hmt["head"]), len(hmt["medium"]), len(hmt["tail"])] == [3, 3, 3]
    assert hmt["head"] == ["cls_0", "cls_1", "cls_2"]
    assert hmt["tail"] == ["cls_6", "cls_7", "cls_8"]
    assert set(freq["group"]) == {"head", "medium", "tail"}
