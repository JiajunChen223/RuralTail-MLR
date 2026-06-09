from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from _bootstrap import bootstrap

bootstrap()

from ruraltail_mlr.data.frequency import compute_frequency_from_files
from ruraltail_mlr.data.label_schema import (
    canonicalize_label_name,
    class_names_from_mapping,
    default_class_mapping,
    write_class_mapping,
)
from ruraltail_mlr.data.split import make_split_from_files


def _find_label_csv(raw_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = raw_root / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(raw_root.glob("*label*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No label CSV found under {raw_root}")
    return candidates[0]


def _find_image_root(raw_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = raw_root / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = [raw_root / "China-MAS-50k_image", raw_root / "images"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    png_parent = next(raw_root.glob("**/*.png"), None)
    if png_parent is None:
        raise FileNotFoundError(f"No PNG images found under {raw_root}")
    return png_parent.parent


def _load_labels(label_csv: Path, class_names: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(label_csv)
    if raw.empty:
        raise ValueError(f"Empty label CSV: {label_csv}")

    first_col = raw.columns[0]
    rename = {first_col: "image_id"}
    for col in raw.columns[1:]:
        canon = canonicalize_label_name(str(col))
        if canon in class_names:
            rename[col] = canon
    labels = raw.rename(columns=rename)

    missing = [name for name in class_names if name not in labels.columns]
    if missing:
        raise ValueError(f"Missing expected label columns after normalization: {missing}")

    keep = ["image_id", *class_names]
    labels = labels[keep].copy()
    labels["image_id"] = labels["image_id"].astype(str)
    labels[class_names] = labels[class_names].astype(int)
    return labels


def _build_images_index(image_root: Path, labels: pd.DataFrame, with_size: bool) -> pd.DataFrame:
    image_files = {path.name: path for path in sorted(image_root.glob("*.png"))}
    rows = []
    for image_id in labels["image_id"].astype(str):
        path = image_files.get(image_id)
        if path is None:
            continue
        row = {"image_id": image_id, "rel_path": path.name}
        if with_size:
            with Image.open(path) as img:
                row["width"], row["height"] = img.size
        rows.append(row)
    if not rows:
        raise ValueError(f"No labeled images matched files in {image_root}")
    return pd.DataFrame(rows)


def prepare(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    mapping = default_class_mapping()
    class_mapping_path = processed_dir / "class_mapping.json"
    write_class_mapping(mapping, class_mapping_path)
    class_names = class_names_from_mapping(mapping)

    label_csv = _find_label_csv(raw_root, args.label_csv)
    image_root = _find_image_root(raw_root, args.image_root)

    labels = _load_labels(label_csv, class_names)
    images = _build_images_index(image_root, labels, with_size=args.with_image_size)
    labels = labels[labels["image_id"].isin(set(images["image_id"]))].reset_index(drop=True)

    images_path = processed_dir / "images_index.csv"
    labels_path = processed_dir / "labels_clean.csv"
    split_path = processed_dir / "fixed_split.csv"
    images.to_csv(images_path, index=False)
    labels.to_csv(labels_path, index=False)

    split = make_split_from_files(labels_path, class_mapping_path, split_path, seed=args.seed)
    compute_frequency_from_files(labels_path, split_path, class_mapping_path, processed_dir)

    print(f"raw_root: {raw_root}")
    print(f"image_root: {image_root}")
    print(f"label_csv: {label_csv}")
    print(f"images: {len(images)}")
    print(f"labels: {len(labels)}")
    print(split["split"].value_counts().to_string())
    print(f"wrote: {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare China-MAS-50k metadata for RuralTail-MLR.")
    parser.add_argument("--raw_root", default="data/raw/china_mas_50k")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--with_image_size", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
