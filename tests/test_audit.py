from pathlib import Path

import pandas as pd
from PIL import Image

from ruraltail_mlr.data.audit import CHINA_MAS_PUBLISHED_ONE_COUNTS, build_label_direction_evidence, run_data_audit
from ruraltail_mlr.data.label_schema import DEFAULT_CLASS_NAMES


def test_audit_marks_label_direction_unverified_by_default(tmp_path: Path):
    raw = tmp_path / "raw"
    image_root = raw / "China-MAS-50k_image"
    image_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(image_root / "image_00001.png")
    row = {"nim": "image_00001.png"}
    for cls in DEFAULT_CLASS_NAMES:
        row[cls.replace("_", " ").title()] = 0
    row["Grassland"] = 1
    pd.DataFrame([row]).to_csv(raw / "China-MAS-50k_label.csv", index=False)
    out = tmp_path / "processed"
    result = run_data_audit(raw, out)
    audit_text = result.audit_md.read_text(encoding="utf-8")
    assert "status: unverified" in audit_text
    assert (out / "label_direction_review.csv").exists()


def test_china_mas_count_consistency_verifies_one_as_present():
    rows = 55520
    labels = pd.DataFrame({"image_id": [f"image_{idx:05d}.png" for idx in range(rows)]})
    for cls in DEFAULT_CLASS_NAMES:
        labels[cls] = 0
        labels.loc[: CHINA_MAS_PUBLISHED_ONE_COUNTS[cls] - 1, cls] = 1
    mapping = {
        "num_classes": len(DEFAULT_CLASS_NAMES),
        "classes": [
            {"idx": idx, "name": name, "short_name": name, "description": name}
            for idx, name in enumerate(DEFAULT_CLASS_NAMES)
        ],
    }

    evidence = build_label_direction_evidence(labels, mapping)

    assert evidence["status"] == "verified_by_published_count_consistency"
    assert evidence["conclusion"] == "raw_value_1_means_class_present"
    assert evidence["one_total_if_present"] == sum(CHINA_MAS_PUBLISHED_ONE_COUNTS.values())
    assert evidence["mean_labels_per_image_if_one_present"] < 3
    assert evidence["mean_labels_per_image_if_zero_present"] > 15
