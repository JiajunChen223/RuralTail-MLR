from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)


def validate_checkpoint_metadata(
    checkpoint: dict,
    class_mapping_hash: str | None = None,
) -> None:
    ckpt_mapping_hash = checkpoint.get("class_mapping_hash")
    if class_mapping_hash is not None and ckpt_mapping_hash is not None and ckpt_mapping_hash != class_mapping_hash:
        raise ValueError(
            "Checkpoint class_mapping_hash does not match current class mapping: "
            f"{ckpt_mapping_hash} != {class_mapping_hash}"
        )
