import numpy as np

from ruraltail_mlr.metrics.hmt import compute_hmt_metrics
from ruraltail_mlr.metrics.multilabel import compute_all_metrics, compute_per_class_ap


def test_multilabel_metrics_small_example():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_prob = np.array([[0.9, 0.2], [0.1, 0.8], [0.6, 0.7]])
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_all_metrics(y_true, y_prob, y_pred)
    assert metrics["mAP"] > 0.9
    assert metrics["OF1"] == 1.0
    per_class = compute_per_class_ap(y_true, y_prob, ["a", "b"])
    assert len(per_class) == 2
    assert {"support", "precision", "recall", "F1"}.issubset(per_class[0])
    hmt = compute_hmt_metrics(y_true, y_prob, y_pred, ["a", "b"], {"head": ["a"], "medium": [], "tail": ["b"]})
    assert hmt["head_mAP"] > 0.9
    assert hmt["tail_F1"] == 1.0
    assert hmt["Tail_mAP"] == hmt["tail_mAP"]
