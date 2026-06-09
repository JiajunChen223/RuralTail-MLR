from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from ruraltail_mlr.data.dataset import ChinaMASDataset
from ruraltail_mlr.data.label_schema import class_names_from_mapping, load_class_mapping, mapping_hash
from ruraltail_mlr.data.transforms import build_transform
from ruraltail_mlr.engine.checkpoint import load_checkpoint, validate_checkpoint_metadata
from ruraltail_mlr.engine.evaluator import Evaluator
from ruraltail_mlr.metrics.threshold import clone_threshold_spec, plain_threshold_grid, summarize_threshold_spec, tune_thresholds_on_val
from ruraltail_mlr.models.factory import build_model
from ruraltail_mlr.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a fixed split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mapping = load_class_mapping(cfg.data.class_mapping)
    class_names = class_names_from_mapping(mapping)
    model = build_model(cfg.model, cfg.data.class_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    validate_checkpoint_metadata(
        ckpt,
        class_mapping_hash=mapping_hash(mapping),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    ds = ChinaMASDataset(
        images_index_csv=cfg.data.images_index,
        labels_csv=cfg.data.labels,
        split_csv=cfg.data.split,
        split=args.split,
        class_mapping_json=cfg.data.class_mapping,
        transform=build_transform(cfg.train.input_size, train=False),
        image_root=cfg.data.image_root,
    )
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=False, num_workers=int(cfg.train.num_workers))
    hmt = json.loads(Path(cfg.data.hmt).read_text(encoding="utf-8")) if Path(cfg.data.hmt).exists() else {}
    evaluator = Evaluator(class_names, hmt, device)
    threshold = clone_threshold_spec(ckpt.get("eval_threshold", cfg.eval.threshold))
    threshold_grid = plain_threshold_grid(cfg.eval.get("threshold_grid", None))
    threshold_strategy = str(cfg.eval.get("threshold_strategy", "global")).lower()
    if bool(cfg.eval.get("tune_threshold_on_val", False)) and "eval_threshold" not in ckpt:
        val_ds = ChinaMASDataset(
            images_index_csv=cfg.data.images_index,
            labels_csv=cfg.data.labels,
            split_csv=cfg.data.split,
            split="val",
            class_mapping_json=cfg.data.class_mapping,
            transform=build_transform(cfg.train.input_size, train=False),
            image_root=cfg.data.image_root,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=int(cfg.train.num_workers),
        )
        val_logits, val_targets, _ = evaluator.predict(model, val_loader)
        threshold, best_score = tune_thresholds_on_val(
            val_targets.numpy().astype(int),
            torch.sigmoid(val_logits).numpy(),
            strategy=threshold_strategy,
            class_names=class_names,
            class_groups=hmt,
            grid=threshold_grid,
            fallback_threshold=float(cfg.eval.threshold),
        )
        threshold_meta = summarize_threshold_spec(threshold, class_names, hmt)
        save_json(
            {
                "split": "val",
                **threshold_meta,
                "macro_F1": float(best_score),
                "grid": threshold_grid,
            },
            Path(args.checkpoint).parent / "threshold_val.json",
        )
    logits, targets, ids = evaluator.predict(model, loader)
    metrics, per_class, hmt_metrics, preds = evaluator.evaluate(logits, targets, ids, threshold=threshold)
    out_dir = Path(args.checkpoint).parent
    save_json(metrics, out_dir / f"metrics_{args.split}.json")
    if args.split == "test":
        import pandas as pd

        pd.DataFrame(per_class).to_csv(out_dir / "per_class_ap_test.csv", index=False)
        save_json(hmt_metrics, out_dir / "hmt_metrics_test.json")
        preds.insert(1, "split", args.split)
        preds.to_csv(out_dir / "predictions_test.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
