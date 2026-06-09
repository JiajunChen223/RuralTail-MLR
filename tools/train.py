from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import hydra
import torch
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ruraltail_mlr.data.dataset import ChinaMASDataset
from ruraltail_mlr.data.label_schema import class_names_from_mapping, load_class_mapping, mapping_hash
from ruraltail_mlr.data.transforms import build_transform
from ruraltail_mlr.engine.distributed import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    per_device_batch_size,
)
from ruraltail_mlr.engine.seed import seed_everything
from ruraltail_mlr.engine.trainer import Trainer
from ruraltail_mlr.losses.factory import build_loss
from ruraltail_mlr.models.factory import build_model
from ruraltail_mlr.utils.config import save_resolved_config
from ruraltail_mlr.utils.io import save_json
from ruraltail_mlr.utils.paths import create_run_dir
from ruraltail_mlr.utils.run_meta import collect_environment_info


def build_loaders(cfg, distributed_state):
    train_tf = build_transform(cfg.train.input_size, train=True)
    eval_tf = build_transform(cfg.train.input_size, train=False)
    common = {
        "images_index_csv": cfg.data.images_index,
        "labels_csv": cfg.data.labels,
        "split_csv": cfg.data.split,
        "class_mapping_json": cfg.data.class_mapping,
        "image_root": cfg.data.image_root,
    }
    train_ds = ChinaMASDataset(split="train", transform=train_tf, **common)
    val_ds = ChinaMASDataset(split="val", transform=eval_tf, **common)
    test_ds = ChinaMASDataset(split="test", transform=eval_tf, **common)
    train_sampler = None
    if distributed_state.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=distributed_state.world_size,
            rank=distributed_state.rank,
            shuffle=True,
            seed=int(cfg.run.seed),
            drop_last=False,
        )
    train_kwargs = {
        "batch_size": per_device_batch_size(int(cfg.train.batch_size), distributed_state),
        "num_workers": int(cfg.train.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    eval_kwargs = {
        "batch_size": int(cfg.train.batch_size),
        "num_workers": int(cfg.train.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **train_kwargs),
        DataLoader(val_ds, shuffle=False, **eval_kwargs),
        DataLoader(test_ds, shuffle=False, **eval_kwargs),
    )


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.train.get("optimizer", "adamw")).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=float(cfg.train.lr),
            weight_decay=float(cfg.train.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    scheduler_cfg = cfg.train.get("scheduler", {})
    scheduler_name = str(scheduler_cfg.get("name", "cosine")).lower()
    if scheduler_name in {"none", "off", "disabled"}:
        return None

    epochs = max(int(cfg.train.epochs), 1)
    eta_min = float(scheduler_cfg.get("eta_min", 0.0))
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    if scheduler_name in {"warmup_cosine", "linear_warmup_cosine"}:
        warmup_epochs = max(int(scheduler_cfg.get("warmup_epochs", 5)), 0)
        warmup_start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.1))
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("train.scheduler.warmup_start_factor must be in (0, 1].")
        if warmup_epochs <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        if warmup_epochs >= epochs:
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=epochs,
            )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    distributed_state = init_distributed(str(cfg.train.device))
    seed_everything(int(cfg.run.seed) + distributed_state.rank)
    try:
        if distributed_state.is_main_process:
            run_dir = create_run_dir(cfg.run.output_dir, cfg.run.name)
            save_resolved_config(cfg, run_dir / "resolved_config.yaml")
            save_json(collect_environment_info("."), run_dir / "environment.json")
        else:
            run_dir = None
        run_dir = Path(broadcast_object(str(run_dir), distributed_state))
        barrier(distributed_state)

        mapping = load_class_mapping(cfg.data.class_mapping)
        class_names = class_names_from_mapping(mapping)
        class_hash = mapping_hash(mapping)
        hmt = {}
        if Path(cfg.data.hmt).exists():
            hmt = json.loads(Path(cfg.data.hmt).read_text(encoding="utf-8"))

        train_loader, val_loader, test_loader = build_loaders(cfg, distributed_state)
        model = build_model(cfg.model, cfg.data.class_mapping)
        model.to(distributed_state.device)
        if distributed_state.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[distributed_state.local_rank] if distributed_state.device.type == "cuda" else None,
                output_device=distributed_state.local_rank if distributed_state.device.type == "cuda" else None,
            )
        criterion = build_loss(cfg.loss)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)

        trainer = Trainer(
            cfg,
            model,
            criterion,
            optimizer,
            scheduler,
            run_dir,
            class_names,
            hmt,
            class_mapping_hash=class_hash,
            device=distributed_state.device,
            distributed_state=distributed_state,
        )
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader, checkpoint="best", val_loader=val_loader)
        if distributed_state.is_main_process:
            print(f"run_dir: {run_dir}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
