from pathlib import Path

import pandas as pd
from PIL import Image

from ruraltail_mlr.data.dataset import ChinaMASDataset
from ruraltail_mlr.data.label_schema import DEFAULT_CLASS_NAMES, default_class_mapping, write_class_mapping
from ruraltail_mlr.data.transforms import build_transform


def test_dataset_returns_required_dict(tmp_path: Path):
    image_root = tmp_path / "China-MAS-50k_image"
    image_root.mkdir()
    Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(image_root / "image_00001.png")

    images_index = tmp_path / "images_index.csv"
    pd.DataFrame(
        [{"image_id": "image_00001.png", "rel_path": "China-MAS-50k_image/image_00001.png", "width": 16, "height": 16}]
    ).to_csv(images_index, index=False)

    labels = {"image_id": ["image_00001.png"]}
    for idx, name in enumerate(DEFAULT_CLASS_NAMES):
        labels[name] = [1 if idx == 0 else 0]
    labels_csv = tmp_path / "labels_clean.csv"
    pd.DataFrame(labels).to_csv(labels_csv, index=False)

    split_csv = tmp_path / "fixed_split.csv"
    pd.DataFrame([{"image_id": "image_00001.png", "split": "train"}]).to_csv(split_csv, index=False)

    mapping_json = tmp_path / "class_mapping.json"
    write_class_mapping(default_class_mapping(), mapping_json)

    ds = ChinaMASDataset(
        images_index_csv=str(images_index),
        labels_csv=str(labels_csv),
        split_csv=str(split_csv),
        split="train",
        class_mapping_json=str(mapping_json),
        transform=build_transform(32),
        image_root=str(tmp_path),
    )
    sample = ds[0]
    assert set(sample) == {"image", "target", "image_id", "rel_path", "meta"}
    assert sample["image"].shape == (3, 32, 32)
    assert sample["target"].shape == (18,)
    assert sample["target"][0].item() == 1.0
