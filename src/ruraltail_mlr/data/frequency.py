from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .label_schema import class_names_from_mapping, load_class_mapping, validate_label_columns


def compute_class_frequency(
    labels_df: pd.DataFrame,
    split_df: pd.DataFrame,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    validate_label_columns(labels_df.columns, class_names)
    merged = labels_df.merge(split_df, on="image_id", how="inner")
    rows = []
    train_n = max(int((merged["split"] == "train").sum()), 1)
    train_counts = merged.loc[merged["split"] == "train", class_names].sum(axis=0)
    rank_order = train_counts.sort_values(ascending=False).index.tolist()
    thirds = np.array_split(np.asarray(rank_order, dtype=object), 3)
    hmt = {
        "head": [str(cls) for cls in thirds[0].tolist()],
        "medium": [str(cls) for cls in thirds[1].tolist()],
        "tail": [str(cls) for cls in thirds[2].tolist()],
        "strategy": "train_frequency_rank_equal_thirds",
    }
    groups = {
        cls: group
        for group, names in hmt.items()
        if group != "strategy"
        for cls in names
    }

    for cls_idx, cls in enumerate(class_names):
        train_count = int(merged.loc[merged["split"] == "train", cls].sum())
        val_count = int(merged.loc[merged["split"] == "val", cls].sum())
        test_count = int(merged.loc[merged["split"] == "test", cls].sum())
        rows.append(
            {
                "class_idx": cls_idx,
                "class_name": cls,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "train_freq": train_count / train_n,
                "rank": rank_order.index(cls) + 1,
                "group": groups[cls],
            }
        )
    freq = pd.DataFrame(rows).sort_values("class_idx").reset_index(drop=True)
    return freq, hmt


def compute_frequency_from_files(
    labels_csv: str | Path,
    split_csv: str | Path,
    class_mapping_json: str | Path,
    out_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    labels = pd.read_csv(labels_csv)
    split = pd.read_csv(split_csv)
    class_names = class_names_from_mapping(load_class_mapping(class_mapping_json))
    freq, hmt = compute_class_frequency(labels, split, class_names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out_dir / "class_frequency.csv", index=False)
    (out_dir / "head_medium_tail.json").write_text(
        json.dumps(hmt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return freq, hmt
