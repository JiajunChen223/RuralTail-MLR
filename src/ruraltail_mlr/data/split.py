from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .label_schema import class_names_from_mapping, load_class_mapping, validate_label_columns


def _choose_split(
    desired_label_counts: np.ndarray,
    desired_sample_counts: np.ndarray,
    label_idx: int | None,
    rng: np.random.Generator,
) -> int:
    if label_idx is None:
        scores = desired_sample_counts.copy()
    else:
        scores = desired_label_counts[:, label_idx].copy()
        max_score = scores.max()
        candidates = np.flatnonzero(np.isclose(scores, max_score))
        if len(candidates) > 1:
            sample_scores = desired_sample_counts[candidates]
            max_sample_score = sample_scores.max()
            candidates = candidates[np.flatnonzero(np.isclose(sample_scores, max_sample_score))]
        return int(rng.choice(candidates))
    max_score = scores.max()
    candidates = np.flatnonzero(np.isclose(scores, max_score))
    return int(rng.choice(candidates))


def iterative_stratification_split(
    labels: np.ndarray,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 20260425,
) -> np.ndarray:
    """Deterministic multi-label iterative stratification.

    This follows the Sechidis-style idea of assigning rare-label samples first:
    repeatedly pick the currently rarest remaining positive label, then assign
    its samples to the split that still needs that label most. All-zero samples
    are assigned at the end by remaining sample quota.
    """
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 2:
        raise ValueError("labels must have shape [N, C]")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")

    rng = np.random.default_rng(seed)
    n, c = labels.shape
    assignment = np.full(n, -1, dtype=np.int64)
    desired_sample_counts = np.asarray(ratios, dtype=np.float64) * n
    desired_label_counts = np.outer(np.asarray(ratios, dtype=np.float64), labels.sum(axis=0))

    remaining = np.ones(n, dtype=bool)
    remaining_pos_counts = labels.sum(axis=0).astype(np.int64)

    while remaining.any() and remaining_pos_counts.max() > 0:
        positive_labels = np.flatnonzero(remaining_pos_counts > 0)
        min_count = remaining_pos_counts[positive_labels].min()
        rare_labels = positive_labels[remaining_pos_counts[positive_labels] == min_count]
        label_idx = int(rng.choice(rare_labels))

        sample_indices = np.flatnonzero(remaining & (labels[:, label_idx] == 1))
        rng.shuffle(sample_indices)
        for sample_idx in sample_indices:
            if not remaining[sample_idx]:
                continue
            split_idx = _choose_split(desired_label_counts, desired_sample_counts, label_idx, rng)
            assignment[sample_idx] = split_idx
            remaining[sample_idx] = False
            desired_sample_counts[split_idx] -= 1.0
            positive = labels[sample_idx].astype(bool)
            desired_label_counts[split_idx, positive] -= 1.0
            remaining_pos_counts[positive] -= 1

    remaining_indices = np.flatnonzero(remaining)
    rng.shuffle(remaining_indices)
    for sample_idx in remaining_indices:
        split_idx = _choose_split(desired_label_counts, desired_sample_counts, None, rng)
        assignment[sample_idx] = split_idx
        desired_sample_counts[split_idx] -= 1.0
        remaining[sample_idx] = False

    if (assignment < 0).any():
        raise AssertionError("Some samples were not assigned to a split")
    return assignment


def make_fixed_split(
    labels_df: pd.DataFrame,
    class_names: list[str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> pd.DataFrame:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
    validate_label_columns(labels_df.columns, class_names)
    labels = labels_df[class_names].astype(int).to_numpy()
    assignment = iterative_stratification_split(
        labels,
        ratios=(train_ratio, val_ratio, test_ratio),
        seed=seed,
    )
    split_names = np.asarray(["train", "val", "test"])

    rows = []
    for idx, image_id in enumerate(labels_df["image_id"].astype(str)):
        rows.append({"image_id": image_id, "split": str(split_names[assignment[idx]])})
    out = pd.DataFrame(rows)
    assert_no_split_overlap(out)
    return out


def assert_no_split_overlap(split_df: pd.DataFrame) -> None:
    if split_df["image_id"].duplicated().any():
        dupes = split_df.loc[split_df["image_id"].duplicated(), "image_id"].head(10).tolist()
        raise AssertionError(f"Duplicate image IDs in split file: {dupes}")
    train_ids = set(split_df.query("split == 'train'").image_id.astype(str))
    val_ids = set(split_df.query("split == 'val'").image_id.astype(str))
    test_ids = set(split_df.query("split == 'test'").image_id.astype(str))
    if not train_ids.isdisjoint(val_ids):
        raise AssertionError("train and val overlap")
    if not train_ids.isdisjoint(test_ids):
        raise AssertionError("train and test overlap")
    if not val_ids.isdisjoint(test_ids):
        raise AssertionError("val and test overlap")


def split_distribution(labels_df: pd.DataFrame, split_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    merged = labels_df.merge(split_df, on="image_id", how="inner")
    rows = []
    for split_name in ["train", "val", "test"]:
        part = merged[merged["split"] == split_name]
        row = {"split": split_name, "num_images": int(len(part))}
        for cls in class_names:
            row[f"{cls}_count"] = int(part[cls].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def make_split_from_files(
    labels_csv: str | Path,
    class_mapping_json: str | Path,
    out_csv: str | Path,
    seed: int,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    class_names = class_names_from_mapping(load_class_mapping(class_mapping_json))
    split_df = make_fixed_split(labels, class_names, seed=seed)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_csv, index=False)
    dist = split_distribution(labels, split_df, class_names)
    dist.to_csv(out_csv.with_name("split_distribution.csv"), index=False)
    meta = {
        "seed": seed,
        "strategy": "iterative_stratification_sechidis_style",
        "counts": split_df["split"].value_counts().to_dict(),
    }
    out_csv.with_name("split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return split_df
