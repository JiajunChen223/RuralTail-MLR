from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, precision_score, recall_score


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_per_class_ap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    threshold: float | list[float] | np.ndarray = 0.5,
) -> list[dict]:
    if np.isscalar(threshold):
        threshold_vec = np.full(y_prob.shape[1], float(threshold), dtype=np.float32)
    else:
        threshold_vec = np.asarray(threshold, dtype=np.float32).reshape(-1)
        if threshold_vec.size != y_prob.shape[1]:
            raise ValueError(f"Expected {y_prob.shape[1]} per-class thresholds, got {threshold_vec.size}")
    y_pred = (y_prob >= threshold_vec[None, :]).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    rows = []
    for idx, name in enumerate(class_names):
        ap = safe_average_precision(y_true[:, idx], y_prob[:, idx])
        rows.append(
            {
                "class_idx": idx,
                "class_name": name,
                "support": int(support[idx]),
                "AP": ap,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "F1": float(f1[idx]),
                "threshold": float(threshold_vec[idx]),
            }
        )
    return rows


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int) if y_pred is None else np.asarray(y_pred).astype(int)
    per_ap = [safe_average_precision(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])]
    valid_ap = [ap for ap in per_ap if not np.isnan(ap)]
    mAP = float(np.mean(valid_ap)) if valid_ap else float("nan")
    sample_precision = precision_score(y_true, y_pred, average="samples", zero_division=0)
    sample_recall = recall_score(y_true, y_pred, average="samples", zero_division=0)
    sample_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)

    mCP = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mCR = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mCF1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    OP = precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    OR = recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    OF1 = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    return {
        "mAP": mAP,
        "mCP": float(mCP),
        "mCR": float(mCR),
        "mCF1": float(mCF1),
        "OP": float(OP),
        "OR": float(OR),
        "OF1": float(OF1),
        "macro_F1": float(mCF1),
        "micro_F1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "sample_precision": float(sample_precision),
        "sample_recall": float(sample_recall),
        "sample_F1": float(sample_f1),
        "num_classes_without_positive": int(sum(np.isnan(ap) for ap in per_ap)),
        "threshold": float(threshold),
    }
