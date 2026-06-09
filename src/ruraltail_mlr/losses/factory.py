from __future__ import annotations

from .asl import AsymmetricLoss
from .bce import BCEWithLogitsLoss
from .combined import CombinedLoss
from .decision import ASLWithTSoftF1Loss
from .focal import SigmoidFocalLoss


def _get(cfg, key: str, default=None):
    return getattr(cfg, key, cfg.get(key, default) if hasattr(cfg, "get") else default)


def build_cls_loss(cls_cfg):
    name = str(_get(cls_cfg, "name", "asl")).lower()
    if name == "bce":
        return BCEWithLogitsLoss()
    if name in {"focal", "sigmoid_focal"}:
        alpha = _get(cls_cfg, "alpha", 0.25)
        if alpha is not None:
            alpha = float(alpha)
        return SigmoidFocalLoss(
            gamma=float(_get(cls_cfg, "gamma", 2.0)),
            alpha=alpha,
            reduction=str(_get(cls_cfg, "reduction", "mean")),
        )
    if name == "asl_soft_f1":
        return ASLWithTSoftF1Loss(
            frequency_csv=_get(cls_cfg, "frequency_csv", "data/processed/class_frequency.csv"),
            lambda_soft_f1=float(_get(cls_cfg, "lambda_soft_f1", 0.2)),
            lambda_tail_recall=float(_get(cls_cfg, "lambda_tail_recall", 0.05)),
            soft_temperature=float(_get(cls_cfg, "soft_temperature", 0.5)),
            threshold_by_group=_get(cls_cfg, "threshold_by_group", None),
            group_weights=_get(cls_cfg, "group_weights", None),
            tail_recall_target=float(_get(cls_cfg, "tail_recall_target", 0.65)),
            gamma_pos=float(_get(cls_cfg, "gamma_pos", 0.0)),
            gamma_neg=float(_get(cls_cfg, "gamma_neg", 4.0)),
            clip=float(_get(cls_cfg, "clip", 0.05)),
            eps=float(_get(cls_cfg, "eps", 1e-8)),
        )
    if name == "asl":
        return AsymmetricLoss(
            gamma_pos=float(_get(cls_cfg, "gamma_pos", 0.0)),
            gamma_neg=float(_get(cls_cfg, "gamma_neg", 4.0)),
            clip=float(_get(cls_cfg, "clip", 0.05)),
            eps=float(_get(cls_cfg, "eps", 1e-8)),
        )
    raise ValueError(f"Unknown cls loss: {name}")


def build_loss(loss_cfg) -> CombinedLoss:
    cls_cfg = _get(loss_cfg, "cls", loss_cfg)
    cls_loss = build_cls_loss(cls_cfg)
    return CombinedLoss(cls_loss=cls_loss)
