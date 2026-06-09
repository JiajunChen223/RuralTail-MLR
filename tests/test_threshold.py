import numpy as np

from ruraltail_mlr.metrics.threshold import apply_thresholds, default_threshold_grid, tune_classwise_thresholds_on_val, tune_global_threshold_on_val, tune_group_thresholds_on_val


def test_default_threshold_grid_matches_paper_protocol():
    assert default_threshold_grid() == [round(0.10 + 0.05 * idx, 2) for idx in range(13)]


def test_threshold_tuning_returns_grid_value():
    y_true_val = np.array([[1, 0], [0, 1], [1, 1]])
    y_prob_val = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.7]])
    thr, score = tune_global_threshold_on_val(y_true_val, y_prob_val, grid=[0.3, 0.5, 0.7])
    assert thr in {0.3, 0.5, 0.7}
    assert score >= 0.0


def test_group_threshold_tuning_returns_group_spec():
    y_true_val = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_prob_val = np.array([[0.55, 0.40], [0.52, 0.30], [0.35, 0.62], [0.20, 0.58]])
    class_names = ["head_cls", "tail_cls"]
    class_groups = {"head": ["head_cls"], "medium": [], "tail": ["tail_cls"]}

    spec, score = tune_group_thresholds_on_val(
        y_true_val,
        y_prob_val,
        class_names=class_names,
        class_groups=class_groups,
        grid=[0.25, 0.5, 0.6],
    )

    assert spec["strategy"] == "group"
    assert spec["thresholds"]["default"] in {0.25, 0.5, 0.6}
    assert spec["thresholds"]["head"] in {0.25, 0.5, 0.6}
    assert spec["thresholds"]["tail"] in {0.25, 0.5, 0.6}
    y_pred = apply_thresholds(y_prob_val, spec, class_names, class_groups)
    assert y_pred.shape == y_true_val.shape
    assert score >= 0.0


def test_classwise_threshold_tuning_returns_per_class_thresholds():
    y_true_val = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_prob_val = np.array([[0.7, 0.3], [0.4, 0.6], [0.65, 0.75], [0.2, 0.25]])

    spec, score = tune_classwise_thresholds_on_val(y_true_val, y_prob_val, grid=[0.3, 0.5, 0.7])

    assert spec["strategy"] == "classwise"
    assert len(spec["thresholds"]) == 2
    assert all(thr in {0.3, 0.5, 0.7} for thr in spec["thresholds"])
    assert len(spec["per_class_F1"]) == 2
    assert score >= 0.0
