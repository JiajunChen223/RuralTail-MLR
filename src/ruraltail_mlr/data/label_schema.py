from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_CLASS_NAMES = [
    "grassland",
    "factory_building",
    "greenhouse",
    "solid_waste",
    "cropland",
    "road",
    "park",
    "river",
    "lake_pond",
    "woodland",
    "basketball_court",
    "bare_land",
    "dust_proof_net",
    "plastic_mulch",
    "photovoltaic",
    "railway",
    "football_field",
    "rural_village",
]

DEFAULT_SHORT_NAMES = [
    "grass",
    "factory",
    "greenhouse",
    "waste",
    "crop",
    "road",
    "park",
    "river",
    "pond",
    "wood",
    "basketball",
    "bare",
    "dustnet",
    "mulch",
    "pv",
    "rail",
    "football",
    "village",
]


def canonicalize_label_name(name: str) -> str:
    text = name.strip().lower()
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def default_class_mapping() -> dict:
    return {
        "num_classes": len(DEFAULT_CLASS_NAMES),
        "classes": [
            {"idx": idx, "name": name, "short_name": short, "description": name.replace("_", " ")}
            for idx, (name, short) in enumerate(zip(DEFAULT_CLASS_NAMES, DEFAULT_SHORT_NAMES))
        ],
    }


def write_default_class_mapping(path: str | Path) -> dict:
    mapping = default_class_mapping()
    write_class_mapping(mapping, path)
    return mapping


def write_class_mapping(mapping: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_class_mapping(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    validate_class_mapping(mapping)
    return mapping


def class_names_from_mapping(mapping: dict) -> list[str]:
    classes = sorted(mapping["classes"], key=lambda item: item["idx"])
    return [item["name"] for item in classes]


def class_to_idx_from_mapping(mapping: dict) -> dict[str, int]:
    return {item["name"]: int(item["idx"]) for item in mapping["classes"]}


def validate_class_mapping(mapping: dict) -> None:
    if mapping.get("num_classes") != len(mapping.get("classes", [])):
        raise ValueError("class_mapping num_classes does not match classes length")
    indices = [int(item["idx"]) for item in mapping["classes"]]
    if indices != list(range(len(indices))):
        raise ValueError("class_mapping indices must be contiguous and start at 0")
    names = [item["name"] for item in mapping["classes"]]
    if len(names) != len(set(names)):
        raise ValueError("class_mapping contains duplicate class names")


def validate_label_columns(columns: Iterable[str], class_names: list[str]) -> None:
    missing = [name for name in class_names if name not in columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")


def mapping_hash(mapping: dict) -> str:
    payload = json.dumps(mapping, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
