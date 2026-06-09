from __future__ import annotations

import pandas as pd
import torch

from ruraltail_mlr.metrics.hmt import compute_hmt_metrics
from ruraltail_mlr.metrics.multilabel import compute_all_metrics, compute_per_class_ap
from ruraltail_mlr.metrics.threshold import apply_thresholds, clone_threshold_spec, resolve_threshold_array, summarize_threshold_spec


class Evaluator:
    def __init__(self, class_names: list[str], class_groups: dict | None = None, device: torch.device | None = None) -> None:
        self.class_names = class_names
        self.class_groups = class_groups or {}
        self.device = device or torch.device("cpu")

    def predict(self, model, loader):
        all_logits, all_targets, all_ids = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_targets.append(batch["target"].cpu())
                all_ids.extend(batch["image_id"])
        return torch.cat(all_logits), torch.cat(all_targets), all_ids

    def evaluate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ids: list[str],
        threshold=0.5,
    ) -> tuple[dict, list[dict], dict, pd.DataFrame]:
        probs = torch.sigmoid(logits).numpy()
        y_true = targets.numpy().astype(int)
        threshold_vec = resolve_threshold_array(threshold, self.class_names, self.class_groups, default=0.5)
        y_pred = apply_thresholds(probs, threshold, self.class_names, self.class_groups)
        threshold_meta = summarize_threshold_spec(threshold, self.class_names, self.class_groups)
        metrics = compute_all_metrics(y_true, probs, y_pred, threshold=float(threshold_meta["threshold"]))
        metrics["threshold_strategy"] = threshold_meta["strategy"]
        metrics["threshold_spec"] = clone_threshold_spec(threshold)
        if threshold_meta["strategy"] != "global":
            metrics["thresholds"] = threshold_meta.get("thresholds")
        per_class = compute_per_class_ap(y_true, probs, self.class_names, threshold=threshold_vec)
        hmt = compute_hmt_metrics(y_true, probs, y_pred, self.class_names, self.class_groups) if self.class_groups else {}

        rows = []
        for row_idx, image_id in enumerate(ids):
            row = {"image_id": image_id}
            for cls_idx, cls in enumerate(self.class_names):
                row[f"y_true_{cls}"] = int(y_true[row_idx, cls_idx])
                row[f"logit_{cls}"] = float(logits[row_idx, cls_idx])
                row[f"prob_{cls}"] = float(probs[row_idx, cls_idx])
                row[f"pred_{cls}"] = int(y_pred[row_idx, cls_idx])
            rows.append(row)
        return metrics, per_class, hmt, pd.DataFrame(rows)
