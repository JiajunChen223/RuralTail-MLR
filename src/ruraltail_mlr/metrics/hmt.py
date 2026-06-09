from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .multilabel import safe_average_precision


def load_hmt(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_hmt_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    hmt: dict,
) -> dict:
    out = {}
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for group in ["head", "medium", "tail"]:
        idxs = [name_to_idx[name] for name in hmt.get(group, []) if name in name_to_idx]
        if not idxs:
            out[f"{group}_mAP"] = float("nan")
            out[f"{group}_F1"] = float("nan")
            continue
        aps = [safe_average_precision(y_true[:, i], y_prob[:, i]) for i in idxs]
        aps = [ap for ap in aps if not np.isnan(ap)]
        out[f"{group}_mAP"] = float(np.mean(aps)) if aps else float("nan")
        out[f"{group}_F1"] = float(f1_score(y_true[:, idxs], y_pred[:, idxs], average="macro", zero_division=0))
    out["Head_mAP"] = out.get("head_mAP", float("nan"))
    out["Medium_mAP"] = out.get("medium_mAP", float("nan"))
    out["Tail_mAP"] = out.get("tail_mAP", float("nan"))
    out["Head_F1"] = out.get("head_F1", float("nan"))
    out["Medium_F1"] = out.get("medium_F1", float("nan"))
    out["Tail_F1"] = out.get("tail_F1", float("nan"))
    return out
