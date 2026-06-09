from __future__ import annotations

import torch

from ruraltail_mlr.data.label_schema import class_names_from_mapping, load_class_mapping

from .backbones import build_backbone
from .multilabel_model import RuralTailMLRModel
from .recent_multilabel import MLMambaClassifier, SFINClassifier


def _get(cfg, key: str, default=None):
    return getattr(cfg, key, cfg.get(key, default) if hasattr(cfg, "get") else default)


def build_model(cfg, class_mapping_json: str) -> torch.nn.Module:
    mapping = load_class_mapping(class_mapping_json)
    num_classes = len(class_names_from_mapping(mapping))
    name = str(_get(cfg, "name", ""))
    if name == "sfin":
        return SFINClassifier(
            num_classes=num_classes,
            backbone_name=str(_get(cfg, "backbone", "resnet18")),
            pretrained=bool(_get(cfg, "pretrained", True)),
            heads=int(_get(cfg, "heads", 4)),
        )
    if name == "mlmamba":
        return MLMambaClassifier(
            num_classes=num_classes,
            backbone_name=str(_get(cfg, "backbone", "resnet18")),
            pretrained=bool(_get(cfg, "pretrained", True)),
            hidden_dim=int(_get(cfg, "hidden_dim", 128)),
        )
    spec = build_backbone(cfg)
    head = str(_get(cfg, "head", "linear"))
    if head != "linear":
        raise ValueError(f"RuralTail-MLR clean code only supports a linear head, got: {head}")
    return RuralTailMLRModel(
        backbone=spec.module,
        feature_dim=spec.feature_dim,
        num_classes=num_classes,
        dropout=float(_get(cfg, "dropout", 0.0)),
    )
