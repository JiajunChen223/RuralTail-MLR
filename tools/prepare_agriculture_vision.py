from __future__ import annotations

import argparse
import os
import json
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from _bootstrap import bootstrap

bootstrap()

from ruraltail_mlr.data.label_schema import canonicalize_label_name, write_class_mapping
from ruraltail_mlr.data.split import assert_no_split_overlap, make_fixed_split, split_distribution


CLASS_DESCRIPTIONS = {
    "double_plant": "double plant",
    "drydown": "drydown",
    "endrow": "endrow",
    "nutrient_deficiency": "nutrient deficiency",
    "planter_skip": "planter skip",
    "storm_damage": "storm damage",
    "water": "water",
    "waterway": "waterway",
    "weed_cluster": "weed cluster",
}


def _image_id_from_path(path: str) -> str:
    return Path(path).stem


def _field_id(image_id: str) -> str:
    return image_id.split("_", 1)[0]


def _mask_is_positive(fp: BinaryIO, min_positive_pixels: int) -> int:
    with Image.open(fp) as mask:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return int(np.count_nonzero(arr) >= min_positive_pixels)


def _mapping_from_classes(class_names: list[str]) -> dict:
    return {
        "num_classes": len(class_names),
        "classes": [
            {
                "idx": idx,
                "name": class_name,
                "short_name": class_name[:12],
                "description": CLASS_DESCRIPTIONS.get(class_name, class_name.replace("_", " ")),
            }
            for idx, class_name in enumerate(class_names)
        ],
    }


def _read_extracted_label_row(args: tuple[str, str, list[str], dict[str, str], int]) -> tuple[dict[str, int | str], str]:
    raw_root_str, image_id, class_names, class_dir_by_name, min_positive_pixels = args
    raw_root = Path(raw_root_str)
    split = image_id.split("::", 1)[0]
    clean_image_id = image_id.split("::", 1)[1]
    row: dict[str, int | str] = {"image_id": clean_image_id}
    for class_name in class_names:
        mask_path = raw_root / split / "labels" / class_dir_by_name[class_name] / f"{clean_image_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        with mask_path.open("rb") as fp:
            row[class_name] = _mask_is_positive(fp, min_positive_pixels)
    return row, split


def _collect_from_extracted(raw_root: Path, min_positive_pixels: int, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    label_root_by_split = {split: raw_root / split / "labels" for split in ("train", "val")}
    class_dir_by_name = {}
    for label_root in label_root_by_split.values():
        if not label_root.exists():
            continue
        for path in label_root.iterdir():
            if path.is_dir():
                class_dir_by_name[canonicalize_label_name(path.name)] = path.name
    class_names = sorted(class_dir_by_name)
    if not class_names:
        raise FileNotFoundError(f"No Agriculture-Vision label folders found under {raw_root}/train|val/labels")

    tasks: list[tuple[str, str, list[str], dict[str, str], int]] = []
    source_split_by_id: dict[str, str] = {}
    for split in ("train", "val"):
        rgb_dir = raw_root / split / "images" / "rgb"
        if not rgb_dir.exists():
            raise FileNotFoundError(rgb_dir)
        for image_path in sorted(rgb_dir.glob("*.jpg")):
            image_id = image_path.stem
            tasks.append((str(raw_root), f"{split}::{image_id}", class_names, class_dir_by_name, min_positive_pixels))
            source_split_by_id[image_id] = split

    labels_rows: list[dict[str, int | str]] = []
    jobs = max(1, int(jobs))
    if jobs == 1:
        for idx, task in enumerate(tasks, start=1):
            row, _ = _read_extracted_label_row(task)
            labels_rows.append(row)
            if idx % 5000 == 0:
                print(f"processed_label_rows={idx}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for idx, (row, _) in enumerate(executor.map(_read_extracted_label_row, tasks, chunksize=64), start=1):
                labels_rows.append(row)
                if idx % 5000 == 0:
                    print(f"processed_label_rows={idx}/{len(tasks)}", flush=True)

    labels = pd.DataFrame(labels_rows).sort_values("image_id").reset_index(drop=True)
    images = _build_images_index(raw_root, labels["image_id"].astype(str), source_split_by_id)
    return labels, images, class_names


def _collect_from_tar(tar_path: Path, min_positive_pixels: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    labels_by_id: dict[str, dict[str, int | str]] = {}
    source_split_by_id: dict[str, str] = {}
    rgb_rel_by_id: dict[str, str] = {}
    class_names: set[str] = set()

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 5 or parts[0] != "Agriculture-Vision-2021":
                continue
            split = parts[1]
            if split not in {"train", "val"}:
                continue
            if parts[2] == "images" and parts[3] == "rgb" and member.name.lower().endswith(".jpg"):
                image_id = _image_id_from_path(member.name)
                rgb_rel_by_id[image_id] = str(Path(*parts[1:])).replace("\\", "/")
                source_split_by_id.setdefault(image_id, split)
                continue
            if parts[2] != "labels" or not member.name.lower().endswith(".png"):
                continue
            class_name = canonicalize_label_name(parts[3])
            class_names.add(class_name)
            image_id = _image_id_from_path(member.name)
            row = labels_by_id.setdefault(image_id, {"image_id": image_id})
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            with extracted:
                row[class_name] = _mask_is_positive(extracted, min_positive_pixels)
            source_split_by_id[image_id] = split

    ordered_classes = sorted(class_names)
    rows: list[dict[str, int | str]] = []
    for image_id, row in labels_by_id.items():
        complete = {"image_id": image_id}
        for class_name in ordered_classes:
            complete[class_name] = int(row.get(class_name, 0))
        rows.append(complete)
    labels = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    missing_rgb = sorted(set(labels["image_id"].astype(str)) - set(rgb_rel_by_id))
    if missing_rgb:
        raise FileNotFoundError(f"Missing {len(missing_rgb)} RGB images in tar, first few: {missing_rgb[:10]}")
    images = pd.DataFrame(
        [{"image_id": image_id, "rel_path": rgb_rel_by_id[image_id]} for image_id in labels["image_id"].astype(str)]
    ).sort_values("image_id").reset_index(drop=True)
    return labels, images, ordered_classes, source_split_by_id


def _extract_rgb_from_tar(tar_path: Path, raw_root: Path, rel_paths: Iterable[str]) -> int:
    wanted = {str(Path(rel_path)).replace("\\", "/") for rel_path in rel_paths}
    extracted_count = 0
    raw_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 5 or parts[0] != "Agriculture-Vision-2021":
                continue
            rel_path = str(Path(*parts[1:])).replace("\\", "/")
            if rel_path not in wanted:
                continue
            out_path = raw_root / rel_path
            if out_path.exists() and out_path.stat().st_size == member.size:
                extracted_count += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member.name)
            with extracted, out_path.open("wb") as fp:
                shutil.copyfileobj(extracted, fp)
            extracted_count += 1
    missing = [rel_path for rel_path in sorted(wanted) if not (raw_root / rel_path).exists()]
    if missing:
        raise FileNotFoundError(f"RGB extraction missed {len(missing)} files, first few: {missing[:10]}")
    return extracted_count


def _build_images_index(raw_root: Path, image_ids: Iterable[str], source_split_by_id: dict[str, str]) -> pd.DataFrame:
    rows = []
    missing = []
    for image_id in sorted(set(image_ids)):
        split = source_split_by_id.get(image_id)
        if split is None:
            missing.append(image_id)
            continue
        rel_path = Path(split) / "images" / "rgb" / f"{image_id}.jpg"
        if not (raw_root / rel_path).exists():
            missing.append(image_id)
            continue
        rows.append({"image_id": image_id, "rel_path": str(rel_path).replace("\\", "/")})
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} Agriculture-Vision RGB images, first few: {missing[:10]}")
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def _make_group_split(labels: pd.DataFrame, class_names: list[str], seed: int, ratios: tuple[float, float, float]) -> pd.DataFrame:
    grouped = labels.copy()
    grouped["field_id"] = grouped["image_id"].astype(str).map(_field_id)
    field_labels = grouped.groupby("field_id", as_index=False)[class_names].max()
    field_labels = field_labels.rename(columns={"field_id": "image_id"})
    field_split = make_fixed_split(
        field_labels,
        class_names,
        seed=seed,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
    )
    split_by_field = dict(zip(field_split["image_id"].astype(str), field_split["split"].astype(str)))
    split_df = pd.DataFrame(
        {
            "image_id": labels["image_id"].astype(str),
            "split": labels["image_id"].astype(str).map(lambda image_id: split_by_field[_field_id(image_id)]),
        }
    )
    assert_no_split_overlap(split_df)
    return split_df


def _write_frequency_equal_thirds(
    labels: pd.DataFrame,
    split_df: pd.DataFrame,
    class_names: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    merged = labels.merge(split_df, on="image_id", how="inner")
    train_n = max(int((merged["split"] == "train").sum()), 1)
    train_counts = merged.loc[merged["split"] == "train", class_names].sum(axis=0)
    rank_order = train_counts.sort_values(ascending=False).index.tolist()
    thirds = np.array_split(np.asarray(rank_order, dtype=object), 3)
    hmt = {
        "head": [str(x) for x in thirds[0].tolist()],
        "medium": [str(x) for x in thirds[1].tolist()],
        "tail": [str(x) for x in thirds[2].tolist()],
        "strategy": "train_frequency_rank_equal_thirds_dataset_local",
    }
    group_by_class = {class_name: group for group, names in hmt.items() if group != "strategy" for class_name in names}
    rows = []
    for class_idx, class_name in enumerate(class_names):
        rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "train_count": int(merged.loc[merged["split"] == "train", class_name].sum()),
                "val_count": int(merged.loc[merged["split"] == "val", class_name].sum()),
                "test_count": int(merged.loc[merged["split"] == "test", class_name].sum()),
                "train_freq": int(merged.loc[merged["split"] == "train", class_name].sum()) / train_n,
                "rank": rank_order.index(class_name) + 1,
                "group": group_by_class[class_name],
            }
        )
    freq = pd.DataFrame(rows).sort_values("class_idx").reset_index(drop=True)
    freq.to_csv(out_dir / "class_frequency.csv", index=False)
    (out_dir / "head_medium_tail.json").write_text(json.dumps(hmt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return freq, hmt


def _parse_ratios(value: str) -> tuple[float, float, float]:
    parts = tuple(float(x) for x in value.split(","))
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError("--ratios must be three comma-separated floats summing to 1")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Agriculture-Vision-2021 image-level metadata for RuralTail-MLR.")
    parser.add_argument("--raw_root", default="data/raw/agriculture_vision_2021/Agriculture-Vision-2021")
    parser.add_argument("--tar_path", default=None, help="Optional Agriculture-Vision-2021.tar.gz for streaming labels.")
    parser.add_argument("--out_dir", default="data/processed/agriculture_vision_2021")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--ratios", type=_parse_ratios, default=(0.8, 0.1, 0.1), help="Grouped train,val,test ratios.")
    parser.add_argument("--min_positive_pixels", type=int, default=1)
    parser.add_argument("--extract_rgb", action="store_true", help="Extract only labeled train/val RGB images from tar_path.")
    parser.add_argument("--jobs", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tar_path:
        tar_path = Path(args.tar_path)
        labels, images, class_names, source_split_by_id = _collect_from_tar(tar_path, args.min_positive_pixels)
        if args.extract_rgb:
            count = _extract_rgb_from_tar(tar_path, raw_root, images["rel_path"].astype(str))
            print(f"extracted_rgb={count} root={raw_root}")
    else:
        labels, images, class_names = _collect_from_extracted(raw_root, args.min_positive_pixels, args.jobs)
        source_split_by_id = {}

    labels_path = out_dir / "labels_clean.csv"
    labels.to_csv(labels_path, index=False)
    images_path = out_dir / "images_index.csv"
    images.to_csv(images_path, index=False)

    mapping = _mapping_from_classes(class_names)
    class_mapping_path = out_dir / "class_mapping.json"
    write_class_mapping(mapping, class_mapping_path)

    split_df = _make_group_split(labels, class_names, args.seed, args.ratios)
    split_path = out_dir / "fixed_split.csv"
    split_df.to_csv(split_path, index=False)
    split_distribution(labels, split_df, class_names).to_csv(out_dir / "split_distribution.csv", index=False)
    _write_frequency_equal_thirds(labels, split_df, class_names, out_dir)

    field_counts = labels["image_id"].astype(str).map(_field_id).nunique()
    meta = {
        "dataset": "Agriculture-Vision-2021",
        "raw_root": str(raw_root),
        "tar_path": args.tar_path,
        "num_images": int(len(labels)),
        "num_fields": int(field_counts),
        "num_classes": int(len(class_names)),
        "classes": class_names,
        "seed": args.seed,
        "ratios": list(args.ratios),
        "min_positive_pixels": int(args.min_positive_pixels),
        "source_official_splits": labels["image_id"].astype(str).map(source_split_by_id).value_counts().to_dict()
        if source_split_by_id
        else None,
        "split_strategy": "grouped_by_field_id_over_official_train_val_labeled_pool",
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"labels_clean: {labels_path}")
    print(f"images_index: {images_path}")
    print(f"class_mapping: {class_mapping_path}")
    print(f"split: {split_path}")
    print(f"class_frequency: {out_dir / 'class_frequency.csv'}")
    print(f"hmt: {out_dir / 'head_medium_tail.json'}")
    print(f"num_images={len(labels)} num_fields={field_counts} num_classes={len(class_names)}")
    print(f"split_counts={split_df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
