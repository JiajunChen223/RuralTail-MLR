from __future__ import annotations

import numpy as np


def paired_difference_summary(a: list[float], b: list[float]) -> dict:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "n": int(len(diff)),
    }
