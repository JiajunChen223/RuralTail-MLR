from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader

from ruraltail_mlr.data.dataset import ChinaMASDataset
from ruraltail_mlr.data.label_schema import DEFAULT_CLASS_NAMES, default_class_mapping, mapping_hash, write_class_mapping
from ruraltail_mlr.data.transforms import build_transform
from ruraltail_mlr.engine.checkpoint import load_checkpoint
from ruraltail_mlr.engine.trainer import Trainer
from ruraltail_mlr.losses.asl import AsymmetricLoss
from ruraltail_mlr.losses.combined import CombinedLoss
from ruraltail_mlr.models.factory import build_model
from ruraltail_mlr.utils.io import load_json


class DictMSELoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux: dict | None = None) -> dict:
        return {"loss": torch.nn.functional.mse_loss(logits, targets)}


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _mini_dataset(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_rows = []
    label_rows = []
    split_rows = []
    for i in range(6):
        image_id = f"image_{i:05d}.png"
        Image.new("RGB", (16, 16), (i * 20, 0, 0)).save(image_root / image_id)
        image_rows.append({"image_id": image_id, "rel_path": image_id, "width": 16, "height": 16})
        row = {"image_id": image_id}
        for cls in DEFAULT_CLASS_NAMES:
            row[cls] = 0
        row[DEFAULT_CLASS_NAMES[i % 3]] = 1
        label_rows.append(row)
        split_rows.append({"image_id": image_id, "split": ["train", "train", "train", "val", "test", "test"][i]})
    images_csv = tmp_path / "images_index.csv"
    labels_csv = tmp_path / "labels.csv"
    split_csv = tmp_path / "split.csv"
    mapping_json = tmp_path / "class_mapping.json"
    pd.DataFrame(image_rows).to_csv(images_csv, index=False)
    pd.DataFrame(label_rows).to_csv(labels_csv, index=False)
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)
    mapping = default_class_mapping()
    write_class_mapping(mapping, mapping_json)
    return images_csv, labels_csv, split_csv, mapping_json, image_root, mapping


def test_one_epoch_smoke_train(tmp_path: Path):
    images_csv, labels_csv, split_csv, mapping_json, image_root, mapping = _mini_dataset(tmp_path)
    transform = build_transform(32)
    common = {
        "images_index_csv": str(images_csv),
        "labels_csv": str(labels_csv),
        "split_csv": str(split_csv),
        "class_mapping_json": str(mapping_json),
        "transform": transform,
        "image_root": str(image_root),
    }
    train_ds = ChinaMASDataset(split="train", **common)
    val_ds = ChinaMASDataset(split="val", **common)
    test_ds = ChinaMASDataset(split="test", **common)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)
    model_cfg = SimpleNamespace(backbone="tiny_cnn", head="linear", embedding_dim=16, dropout=0.0)
    model = build_model(model_cfg, str(mapping_json))
    cfg = SimpleNamespace(
        train=SimpleNamespace(device="cpu", amp=False, epochs=1),
        run=SimpleNamespace(monitor="mAP", monitor_mode="max"),
        eval=SimpleNamespace(threshold=0.5),
    )
    criterion = CombinedLoss(AsymmetricLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        cfg,
        model,
        criterion,
        optimizer,
        scheduler=None,
        run_dir=tmp_path / "run",
        class_names=DEFAULT_CLASS_NAMES,
        class_groups={"head": DEFAULT_CLASS_NAMES[:6], "medium": DEFAULT_CLASS_NAMES[6:12], "tail": DEFAULT_CLASS_NAMES[12:]},
        class_mapping_hash=mapping_hash(mapping),
    )
    trainer.fit(train_loader, val_loader)
    trainer.test(test_loader)
    assert (tmp_path / "run" / "checkpoint_best.pth").exists()
    assert (tmp_path / "run" / "metrics_test.json").exists()
    metrics_test = load_json(tmp_path / "run" / "metrics_test.json")
    ckpt = load_checkpoint(tmp_path / "run" / "checkpoint_best.pth", map_location="cpu")
    assert metrics_test["threshold"] == pytest.approx(0.5)
    assert metrics_test["threshold_strategy"] == "global"
    assert "thresholds" not in metrics_test
    assert "eval" + "_cal" + "ibration" not in ckpt
    assert "best_eval" + "_cal" + "ibration" not in ckpt


def test_early_stopping_respects_min_epochs_and_patience(tmp_path: Path):
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            device="cpu",
            amp=False,
            epochs=10,
            early_stopping=SimpleNamespace(
                enabled=True,
                monitor="mAP",
                mode="max",
                patience=2,
                min_delta=0.01,
                min_epochs=3,
            ),
        ),
        run=SimpleNamespace(monitor="mAP", monitor_mode="max"),
        eval=SimpleNamespace(threshold=0.5),
    )
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(
        cfg,
        model,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        run_dir=tmp_path / "run",
        class_names=["x"],
        class_groups={},
    )

    assert not trainer._update_early_stopping(1, {"mAP": 0.5})["should_stop"]
    assert not trainer._update_early_stopping(2, {"mAP": 0.505})["should_stop"]
    state = trainer._update_early_stopping(3, {"mAP": 0.506})

    assert state["wait"] == 2
    assert state["should_stop"]


def test_gradient_accumulation_steps_optimizer_every_n_batches(tmp_path: Path):
    loader = DataLoader(
        [
            {"image": torch.tensor([[1.0]]), "target": torch.tensor([[0.0]])},
            {"image": torch.tensor([[2.0]]), "target": torch.tensor([[0.0]])},
            {"image": torch.tensor([[3.0]]), "target": torch.tensor([[0.0]])},
        ],
        batch_size=1,
    )
    cfg = SimpleNamespace(
        train=SimpleNamespace(device="cpu", amp=False, epochs=1, gradient_accumulation_steps=2),
        run=SimpleNamespace(monitor="mAP", monitor_mode="max"),
        eval=SimpleNamespace(threshold=0.5),
    )
    model = torch.nn.Linear(1, 1)
    optimizer = CountingSGD(model.parameters(), lr=1e-3)
    trainer = Trainer(
        cfg,
        model,
        criterion=DictMSELoss(),
        optimizer=optimizer,
        scheduler=None,
        run_dir=tmp_path / "run",
        class_names=["x"],
        class_groups={},
    )

    trainer.train_one_epoch(loader, epoch=1)

    assert optimizer.step_calls == 2


def test_val_threshold_tuning_is_persisted_to_checkpoint(tmp_path: Path):
    images_csv, labels_csv, split_csv, mapping_json, image_root, mapping = _mini_dataset(tmp_path)
    transform = build_transform(32)
    common = {
        "images_index_csv": str(images_csv),
        "labels_csv": str(labels_csv),
        "split_csv": str(split_csv),
        "class_mapping_json": str(mapping_json),
        "transform": transform,
        "image_root": str(image_root),
    }
    train_ds = ChinaMASDataset(split="train", **common)
    val_ds = ChinaMASDataset(split="val", **common)
    test_ds = ChinaMASDataset(split="test", **common)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)
    model_cfg = SimpleNamespace(backbone="tiny_cnn", head="linear", embedding_dim=16, dropout=0.0)
    model = build_model(model_cfg, str(mapping_json))
    cfg = SimpleNamespace(
        train=SimpleNamespace(device="cpu", amp=False, epochs=1),
        run=SimpleNamespace(monitor="mAP", monitor_mode="max"),
        eval=SimpleNamespace(threshold=0.5, tune_threshold_on_val=True, threshold_grid=[0.3, 0.5, 0.7]),
    )
    criterion = CombinedLoss(AsymmetricLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_dir = tmp_path / "run"
    trainer = Trainer(
        cfg,
        model,
        criterion,
        optimizer,
        scheduler=None,
        run_dir=run_dir,
        class_names=DEFAULT_CLASS_NAMES,
        class_groups={"head": DEFAULT_CLASS_NAMES[:6], "medium": DEFAULT_CLASS_NAMES[6:12], "tail": DEFAULT_CLASS_NAMES[12:]},
        class_mapping_hash=mapping_hash(mapping),
    )
    trainer.fit(train_loader, val_loader)
    trainer.test(test_loader, val_loader=val_loader)

    threshold_meta = load_json(run_dir / "threshold_val.json")
    ckpt = load_checkpoint(run_dir / "checkpoint_best.pth", map_location="cpu")
    metrics_test = load_json(run_dir / "metrics_test.json")

    assert threshold_meta["threshold"] in {0.3, 0.5, 0.7}
    assert float(ckpt["eval_threshold"]) == pytest.approx(threshold_meta["threshold"])
    assert float(metrics_test["threshold"]) == pytest.approx(threshold_meta["threshold"])
    assert "eval" + "_cal" + "ibration" not in ckpt
    assert "best_eval" + "_cal" + "ibration" not in ckpt


def test_group_threshold_tuning_is_persisted(tmp_path: Path):
    images_csv, labels_csv, split_csv, mapping_json, image_root, mapping = _mini_dataset(tmp_path)
    transform = build_transform(32)
    common = {
        "images_index_csv": str(images_csv),
        "labels_csv": str(labels_csv),
        "split_csv": str(split_csv),
        "class_mapping_json": str(mapping_json),
        "transform": transform,
        "image_root": str(image_root),
    }
    train_ds = ChinaMASDataset(split="train", **common)
    val_ds = ChinaMASDataset(split="val", **common)
    test_ds = ChinaMASDataset(split="test", **common)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)
    model_cfg = SimpleNamespace(backbone="tiny_cnn", head="linear", embedding_dim=16, dropout=0.0)
    model = build_model(model_cfg, str(mapping_json))
    cfg = SimpleNamespace(
        train=SimpleNamespace(device="cpu", amp=False, epochs=1),
        run=SimpleNamespace(monitor="mAP", monitor_mode="max"),
        eval=SimpleNamespace(
            threshold=0.5,
            tune_threshold_on_val=True,
            threshold_strategy="group",
            threshold_grid=[0.3, 0.5, 0.7],
        ),
    )
    criterion = CombinedLoss(AsymmetricLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    run_dir = tmp_path / "run_group"
    trainer = Trainer(
        cfg,
        model,
        criterion,
        optimizer,
        scheduler=None,
        run_dir=run_dir,
        class_names=DEFAULT_CLASS_NAMES,
        class_groups={"head": DEFAULT_CLASS_NAMES[:6], "medium": DEFAULT_CLASS_NAMES[6:12], "tail": DEFAULT_CLASS_NAMES[12:]},
        class_mapping_hash=mapping_hash(mapping),
    )
    trainer.fit(train_loader, val_loader)
    trainer.test(test_loader, val_loader=val_loader)

    threshold_meta = load_json(run_dir / "threshold_val.json")
    ckpt = load_checkpoint(run_dir / "checkpoint_best.pth", map_location="cpu")
    metrics_test = load_json(run_dir / "metrics_test.json")

    assert threshold_meta["strategy"] == "group"
    assert set(threshold_meta["thresholds"]).issuperset({"default"})
    assert isinstance(ckpt["eval_threshold"], dict)
    assert ckpt["eval_threshold"]["strategy"] == "group"
    assert metrics_test["threshold_strategy"] == "group"
