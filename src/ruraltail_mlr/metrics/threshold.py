from __future__ import annotations

from copy import deepcopy
from numbers import Number

import numpy as np
from sklearn.metrics import f1_score


def default_threshold_grid() -> list[float]:
    return [round(0.10 + 0.05 * idx, 2) for idx in range(13)]


def plain_threshold_grid(grid: list[float] | tuple[float, ...] | None) -> list[float] | None:
    if grid is None:
        return None
    return [float(x) for x in grid]


def is_scalar_threshold(threshold) -> bool:
    return isinstance(threshold, Number)


def clone_threshold_spec(threshold):
    if isinstance(threshold, np.ndarray):
        return threshold.astype(float).tolist()
    return deepcopy(threshold)


def _threshold_vector_from_sequence(threshold, num_classes: int) -> np.ndarray:
    vector = np.asarray(threshold, dtype=np.float32).reshape(-1)
    if vector.size != num_classes:
        raise ValueError(f"Expected {num_classes} thresholds, got {vector.size}")
    return vector


def resolve_threshold_array(
    threshold,
    class_names: list[str],
    class_groups: dict | None = None,
    default: float = 0.5,
) -> np.ndarray:
    num_classes = len(class_names)
    if threshold is None:
        return np.full(num_classes, float(default), dtype=np.float32)
    if is_scalar_threshold(threshold):
        return np.full(num_classes, float(threshold), dtype=np.float32)
    if isinstance(threshold, (list, tuple, np.ndarray)):
        return _threshold_vector_from_sequence(threshold, num_classes)
    if not isinstance(threshold, dict):
        raise TypeError(f"Unsupported threshold type: {type(threshold)!r}")

    if "per_class" in threshold:
        return _threshold_vector_from_sequence(threshold["per_class"], num_classes)
    nested_thresholds = threshold.get("thresholds")
    if isinstance(nested_thresholds, (list, tuple, np.ndarray)):
        return _threshold_vector_from_sequence(nested_thresholds, num_classes)

    class_groups = class_groups or {}
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    base = float(threshold.get("default", threshold.get("threshold", default)))
    vector = np.full(num_classes, base, dtype=np.float32)

    group_thresholds = nested_thresholds if isinstance(nested_thresholds, dict) else threshold
    for group_name, value in group_thresholds.items():
        if group_name in {"strategy", "threshold", "default", "macro_F1", "per_group_macro_F1", "per_class_F1", "grid"}:
            continue
        if group_name not in class_groups:
            continue
        thr = float(value)
        for class_name in class_groups.get(group_name, []):
            if class_name in name_to_idx:
                vector[name_to_idx[class_name]] = thr
    return vector


def apply_thresholds(
    y_prob: np.ndarray,
    threshold,
    class_names: list[str] | None = None,
    class_groups: dict | None = None,
) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if is_scalar_threshold(threshold):
        return (y_prob >= float(threshold)).astype(int)
    if class_names is None:
        class_names = [str(idx) for idx in range(y_prob.shape[1])]
    vector = resolve_threshold_array(threshold, class_names, class_groups)
    return (y_prob >= vector[None, :]).astype(int)


def summarize_threshold_spec(
    threshold,
    class_names: list[str] | None = None,
    class_groups: dict | None = None,
) -> dict:
    if threshold is None:
        return {"strategy": "global", "threshold": 0.5}
    if is_scalar_threshold(threshold):
        return {"strategy": "global", "threshold": float(threshold)}
    if class_names is None:
        if isinstance(threshold, dict):
            return clone_threshold_spec(threshold)
        vector = np.asarray(threshold, dtype=np.float32).reshape(-1)
        return {
            "strategy": "classwise",
            "threshold": float(vector.mean()),
            "thresholds": vector.astype(float).tolist(),
        }

    vector = resolve_threshold_array(threshold, class_names, class_groups)
    if isinstance(threshold, dict):
        strategy = str(threshold.get("strategy", "group"))
        summary = clone_threshold_spec(threshold)
        summary["strategy"] = strategy
        summary["threshold"] = float(summary.get("threshold", vector.mean()))
        if strategy == "group":
            summary.setdefault("thresholds", {})
        else:
            summary["thresholds"] = vector.astype(float).tolist()
        return summary

    return {
        "strategy": "classwise",
        "threshold": float(vector.mean()),
        "thresholds": vector.astype(float).tolist(),
    }


def tune_global_threshold_on_val(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    grid: list[float] | None = None,
) -> tuple[float, float]:
    grid = plain_threshold_grid(grid) or default_threshold_grid()
    best_thr = 0.5
    best_score = -1.0
    for thr in grid:
        y_pred = (y_prob_val >= thr).astype(int)
        score = f1_score(y_true_val, y_pred, average="macro", zero_division=0)
        if score > best_score:
            best_thr = float(thr)
            best_score = float(score)
    return best_thr, best_score


def tune_group_thresholds_on_val(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    class_names: list[str],
    class_groups: dict | None,
    grid: list[float] | None = None,
    fallback_threshold: float | None = None,
) -> tuple[dict, float]:
    grid = plain_threshold_grid(grid) or default_threshold_grid()
    class_groups = class_groups or {}
    default_thr, _ = tune_global_threshold_on_val(y_true_val, y_prob_val, grid=grid)
    if fallback_threshold is not None:
        default_thr = float(fallback_threshold)

    thresholds = {"default": float(default_thr)}
    per_group_macro_f1 = {}
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for group_name in ["head", "medium", "tail"]:
        idxs = [name_to_idx[name] for name in class_groups.get(group_name, []) if name in name_to_idx]
        if not idxs:
            continue
        best_thr = float(default_thr)
        best_score = -1.0
        for thr in grid:
            y_pred = (y_prob_val[:, idxs] >= thr).astype(int)
            score = f1_score(y_true_val[:, idxs], y_pred, average="macro", zero_division=0)
            if score > best_score:
                best_thr = float(thr)
                best_score = float(score)
        thresholds[group_name] = best_thr
        per_group_macro_f1[group_name] = best_score

    spec = {
        "strategy": "group",
        "threshold": float(default_thr),
        "thresholds": thresholds,
        "per_group_macro_F1": per_group_macro_f1,
    }
    y_pred_full = apply_thresholds(y_prob_val, spec, class_names, class_groups)
    macro_f1 = float(f1_score(y_true_val, y_pred_full, average="macro", zero_division=0))
    spec["macro_F1"] = macro_f1
    return spec, macro_f1


def tune_classwise_thresholds_on_val(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    grid: list[float] | None = None,
) -> tuple[dict, float]:
    grid = plain_threshold_grid(grid) or default_threshold_grid()
    thresholds = []
    per_class_f1 = []
    for class_idx in range(y_true_val.shape[1]):
        best_thr = 0.5
        best_score = -1.0
        for thr in grid:
            y_pred = (y_prob_val[:, class_idx] >= thr).astype(int)
            score = f1_score(y_true_val[:, class_idx], y_pred, average="binary", zero_division=0)
            if score > best_score:
                best_thr = float(thr)
                best_score = float(score)
        thresholds.append(best_thr)
        per_class_f1.append(best_score)
    threshold_array = np.asarray(thresholds, dtype=np.float32)
    y_pred_full = (y_prob_val >= threshold_array[None, :]).astype(int)
    macro_f1 = float(f1_score(y_true_val, y_pred_full, average="macro", zero_division=0))
    spec = {
        "strategy": "classwise",
        "threshold": float(threshold_array.mean()),
        "thresholds": threshold_array.astype(float).tolist(),
        "per_class_F1": [float(x) for x in per_class_f1],
        "macro_F1": macro_f1,
    }
    return spec, macro_f1


def tune_thresholds_on_val(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    strategy: str = "global",
    class_names: list[str] | None = None,
    class_groups: dict | None = None,
    grid: list[float] | None = None,
    fallback_threshold: float = 0.5,
):
    strategy = str(strategy).lower()
    if strategy == "global":
        thr, score = tune_global_threshold_on_val(y_true_val, y_prob_val, grid=grid)
        return float(thr), float(score)
    if strategy == "group":
        if class_names is None:
            raise ValueError("class_names are required for group threshold tuning")
        return tune_group_thresholds_on_val(
            y_true_val,
            y_prob_val,
            class_names=class_names,
            class_groups=class_groups,
            grid=grid,
            fallback_threshold=fallback_threshold,
        )
    if strategy == "classwise":
        return tune_classwise_thresholds_on_val(y_true_val, y_prob_val, grid=grid)
    raise ValueError(f"Unknown threshold strategy: {strategy}")
