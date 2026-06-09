from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from .label_schema import (
    canonicalize_label_name,
    class_names_from_mapping,
    default_class_mapping,
    validate_label_columns,
    write_class_mapping,
)


@dataclass
class AuditResult:
    labels_clean: Path
    images_index: Path
    class_mapping: Path
    audit_md: Path
    label_direction_evidence: Path | None = None


CHINA_MAS_PUBLISHED_ONE_COUNTS = {
    "grassland": 1492,
    "factory_building": 5979,
    "greenhouse": 1874,
    "solid_waste": 964,
    "cropland": 29958,
    "road": 14959,
    "park": 1222,
    "river": 5451,
    "lake_pond": 6078,
    "woodland": 36832,
    "basketball_court": 1453,
    "bare_land": 7093,
    "dust_proof_net": 471,
    "plastic_mulch": 920,
    "photovoltaic": 755,
    "railway": 2156,
    "football_field": 1683,
    "rural_village": 15947,
}


def build_label_direction_evidence(labels: pd.DataFrame, mapping: dict, tolerance: int = 2) -> dict:
    class_names = class_names_from_mapping(mapping)
    validate_label_columns(labels.columns, class_names)
    one_counts = labels[class_names].sum(axis=0).astype(int)
    zero_counts = (len(labels) - one_counts).astype(int)

    comparable = [cls for cls in class_names if cls in CHINA_MAS_PUBLISHED_ONE_COUNTS]
    deltas = {
        cls: int(one_counts[cls] - CHINA_MAS_PUBLISHED_ONE_COUNTS[cls])
        for cls in comparable
    }
    max_abs_delta = max((abs(delta) for delta in deltas.values()), default=None)
    matches_published_counts = (
        len(comparable) == len(class_names)
        and max_abs_delta is not None
        and max_abs_delta <= tolerance
    )

    conclusion = "unverified"
    status = "unverified"
    if matches_published_counts:
        status = "verified_by_published_count_consistency"
        conclusion = "raw_value_1_means_class_present"

    return {
        "status": status,
        "conclusion": conclusion,
        "evidence_basis": (
            "The raw value-1 class counts match the published China-MAS-50k class "
            "counts within tolerance. Interpreting raw value 0 as present would imply "
            "the complement counts and an implausibly dense multi-label target."
        ),
        "note": (
            "The Nature Scientific Data article contains a contradictory Data Records "
            "sentence about 0/1 semantics. Count consistency and the official baseline "
            "usage support preserving the raw direction as 1=present, 0=absent."
        ),
        "tolerance": tolerance,
        "num_images": int(len(labels)),
        "one_total_if_present": int(one_counts.sum()),
        "zero_total_if_present": int(zero_counts.sum()),
        "mean_labels_per_image_if_one_present": float(one_counts.sum() / len(labels)),
        "mean_labels_per_image_if_zero_present": float(zero_counts.sum() / len(labels)),
        "published_one_counts": {
            cls: int(CHINA_MAS_PUBLISHED_ONE_COUNTS[cls]) for cls in comparable
        },
        "observed_one_counts": {cls: int(one_counts[cls]) for cls in class_names},
        "observed_zero_counts": {cls: int(zero_counts[cls]) for cls in class_names},
        "one_count_minus_published": deltas,
        "max_abs_delta": max_abs_delta,
    }


def find_raw_labels(raw_root: str | Path) -> Path:
    raw_root = Path(raw_root)
    candidates = sorted(raw_root.glob("*label*.csv")) + sorted(raw_root.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No label CSV found under {raw_root}")
    return candidates[0]


def find_image_root(raw_root: str | Path) -> Path:
    raw_root = Path(raw_root)
    candidates = [p for p in raw_root.iterdir() if p.is_dir() and "image" in p.name.lower()]
    if candidates:
        return candidates[0]
    return raw_root


def normalize_labels(raw_label_csv: str | Path, out_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_label_csv)
    if raw.empty:
        raise ValueError("Raw label CSV is empty")

    id_col = raw.columns[0]
    label_cols = list(raw.columns[1:])
    clean_names = [canonicalize_label_name(col) for col in label_cols]

    df = pd.DataFrame()
    df["image_id"] = raw[id_col].astype(str)
    for raw_col, clean_col in zip(label_cols, clean_names):
        df[clean_col] = raw[raw_col]

    values = df[clean_names].to_numpy()
    invalid = sorted(set(values.ravel()) - {0, 1, "0", "1"})
    if invalid:
        raise ValueError(f"Labels must be binary 0/1; found {invalid[:10]}")
    df[clean_names] = df[clean_names].astype(int)

    mapping = {
        "num_classes": len(clean_names),
        "classes": [
            {
                "idx": idx,
                "name": name,
                "short_name": name.replace("_", "")[:12],
                "description": raw_col,
            }
            for idx, (name, raw_col) in enumerate(zip(clean_names, label_cols))
        ],
    }
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, mapping


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_images_index(
    image_root: str | Path,
    image_ids: list[str],
    out_csv: str | Path,
    check_images: bool = True,
    checksum: bool = False,
) -> pd.DataFrame:
    image_root = Path(image_root)
    rows = []
    missing = []
    corrupt = []
    for image_id in image_ids:
        path = image_root / image_id
        rel_path = str(path.relative_to(image_root.parent)) if path.exists() else image_id
        width = height = None
        mode = None
        digest = ""
        if not path.exists():
            missing.append(image_id)
        elif check_images:
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    mode = img.mode
                if checksum:
                    digest = file_sha256(path)
            except Exception:
                corrupt.append(image_id)
        rows.append(
            {
                "image_id": image_id,
                "rel_path": rel_path.replace("\\", "/"),
                "width": width,
                "height": height,
                "mode": mode,
                "checksum": digest,
            }
        )
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if missing:
        pd.DataFrame({"image_id": missing}).to_csv(out_csv.parent / "missing_images.csv", index=False)
    if corrupt:
        pd.DataFrame({"image_id": corrupt}).to_csv(out_csv.parent / "corrupted_images.csv", index=False)
    return df


def write_audit_md(
    out_path: str | Path,
    labels: pd.DataFrame,
    mapping: dict,
    images_index: pd.DataFrame,
    raw_label_csv: Path,
    image_root: Path,
    label_direction_verified: bool = False,
    label_direction_note: str = "",
    label_direction_evidence: dict | None = None,
) -> None:
    class_names = class_names_from_mapping(mapping)
    validate_label_columns(labels.columns, class_names)
    counts = labels[class_names].sum(axis=0).sort_values(ascending=False)
    empty_rows = int((labels[class_names].sum(axis=1) == 0).sum())
    missing_count = int(images_index["width"].isna().sum()) if "width" in images_index else 0
    modes = images_index["mode"].dropna().value_counts().to_dict() if "mode" in images_index else {}
    evidence_status = (label_direction_evidence or {}).get("status", "unverified")
    evidence_conclusion = (label_direction_evidence or {}).get("conclusion", "unverified")
    evidence_verified = str(evidence_status).startswith("verified")
    direction_status = "verified" if label_direction_verified or evidence_verified else "unverified"
    if direction_status == "verified":
        direction_statement = (
            "The processed labels preserve the raw CSV direction: 1 means class present, "
            "0 means class absent."
        )
    else:
        direction_statement = (
            "The script only verified binary 0/1 values and preserved the raw direction. "
            "The semantic meaning of 1 is not automatically verified."
        )
    lines = [
        "# Data Audit",
        "",
        f"- raw_label_csv: `{raw_label_csv}`",
        f"- image_root: `{image_root}`",
        f"- images: {len(labels)}",
        f"- classes: {len(class_names)}",
        f"- total_positive_labels: {int(counts.sum())}",
        f"- empty_label_images: {empty_rows}",
        f"- unread_or_missing_images: {missing_count}",
        f"- image_modes: {modes}",
        "",
        "## Label Direction",
        "",
        f"- status: {direction_status}",
        f"- note: {label_direction_note or 'not provided'}",
        f"- evidence_status: {evidence_status}",
        f"- evidence_conclusion: {evidence_conclusion}",
        direction_statement,
    ]
    if label_direction_evidence:
        lines.extend(
            [
                f"- one_total_if_present: {label_direction_evidence['one_total_if_present']}",
                f"- zero_total_if_present: {label_direction_evidence['zero_total_if_present']}",
                f"- evidence_basis: {label_direction_evidence['evidence_basis']}",
                f"- caution: {label_direction_evidence['note']}",
            ]
        )
    if direction_status != "verified":
        lines.append(
            "Manual visual spot checks and/or official schema evidence must be recorded here before final experiments."
        )
    lines.extend(
        [
            "",
            "## Class Counts",
            "",
            "| class | count | frequency |",
            "|---|---:|---:|",
        ]
    )
    for cls, count in counts.items():
        lines.append(f"| {cls} | {int(count)} | {count / len(labels):.6f} |")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_data_audit(
    raw_root: str | Path,
    out_dir: str | Path,
    check_images: bool = True,
    checksum: bool = False,
    label_direction_verified: bool = False,
    label_direction_note: str = "",
) -> AuditResult:
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_label_csv = find_raw_labels(raw_root)
    image_root = find_image_root(raw_root)

    labels_clean = out_dir / "labels_clean.csv"
    labels, mapping = normalize_labels(raw_label_csv, labels_clean)
    if mapping["num_classes"] != default_class_mapping()["num_classes"]:
        raise ValueError("China-MAS-50k should contain exactly 18 classes")

    class_mapping = out_dir / "class_mapping.json"
    write_class_mapping(mapping, class_mapping)
    images_index = out_dir / "images_index.csv"
    images_df = build_images_index(
        image_root=image_root,
        image_ids=labels["image_id"].tolist(),
        out_csv=images_index,
        check_images=check_images,
        checksum=checksum,
    )
    audit_md = out_dir / "data_audit.md"
    evidence = build_label_direction_evidence(labels, mapping)
    evidence_path = out_dir / "label_direction_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_label_direction_review(labels, mapping, out_dir / "label_direction_review.csv")
    write_audit_md(
        audit_md,
        labels,
        mapping,
        images_df,
        raw_label_csv,
        image_root,
        label_direction_verified=label_direction_verified,
        label_direction_note=label_direction_note,
        label_direction_evidence=evidence,
    )
    return AuditResult(labels_clean, images_index, class_mapping, audit_md, evidence_path)


def write_label_direction_review(labels: pd.DataFrame, mapping: dict, out_csv: str | Path) -> None:
    class_names = class_names_from_mapping(mapping)
    rows = []
    for cls in class_names:
        positives = labels.loc[labels[cls] == 1, "image_id"].astype(str).head(5).tolist()
        negatives = labels.loc[labels[cls] == 0, "image_id"].astype(str).head(5).tolist()
        rows.append(
            {
                "class_name": cls,
                "positive_examples": ";".join(positives),
                "negative_examples": ";".join(negatives),
                "manual_status": "unverified",
                "review_note": "",
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
