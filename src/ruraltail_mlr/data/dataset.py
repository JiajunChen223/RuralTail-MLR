from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_schema import class_names_from_mapping, load_class_mapping, validate_label_columns


class ChinaMASDataset(Dataset):
    def __init__(
        self,
        images_index_csv: str,
        labels_csv: str,
        split_csv: str,
        split: Literal["train", "val", "test"],
        class_mapping_json: str,
        transform: Callable | None,
        return_meta: bool = False,
        image_root: str | None = None,
    ) -> None:
        self.images_index_csv = Path(images_index_csv)
        self.labels_csv = Path(labels_csv)
        self.split_csv = Path(split_csv)
        self.split = split
        self.mapping = load_class_mapping(class_mapping_json)
        self.class_names = class_names_from_mapping(self.mapping)
        self.transform = transform
        self.return_meta = return_meta
        self.image_root = Path(image_root) if image_root else self.images_index_csv.parent.parent

        labels = pd.read_csv(self.labels_csv)
        validate_label_columns(labels.columns, self.class_names)
        images = pd.read_csv(self.images_index_csv)
        split_df = pd.read_csv(self.split_csv)

        if "image_id" not in split_df or "split" not in split_df:
            raise ValueError("split_csv must contain image_id and split columns")

        split_ids = split_df.loc[split_df["split"] == split, "image_id"].astype(str)
        merged = (
            labels.assign(image_id=labels["image_id"].astype(str))
            .merge(images.assign(image_id=images["image_id"].astype(str)), on="image_id", how="inner")
            .merge(pd.DataFrame({"image_id": split_ids}), on="image_id", how="inner")
        )
        if merged.empty:
            raise ValueError(f"No rows found for split={split}")
        self.df = merged.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_path: str) -> Path:
        path = Path(rel_path)
        if path.is_absolute():
            return path
        candidates = [
            self.image_root / rel_path,
            self.image_root.parent / rel_path,
            self.images_index_csv.parent.parent / rel_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        image_id = str(row["image_id"])
        rel_path = str(row.get("rel_path", image_id))
        image_path = self._resolve_image_path(rel_path)
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        target = torch.tensor(row[self.class_names].astype("float32").to_numpy(), dtype=torch.float32)
        sample = {
            "image": image_tensor,
            "target": target,
            "image_id": image_id,
            "rel_path": rel_path,
            "meta": {},
        }
        if self.return_meta:
            sample["meta"] = row.to_dict()
        return sample
