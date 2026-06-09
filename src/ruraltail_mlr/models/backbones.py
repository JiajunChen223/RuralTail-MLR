from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

@dataclass
class BackboneSpec:
    module: nn.Module
    feature_dim: int


class TinyBackbone(nn.Module):
    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def _get(cfg, key: str, default=None):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _as_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "items"):
        return {str(k): v for k, v in value.items()}
    raise TypeError("timm_kwargs must be a mapping")


def build_backbone(cfg) -> BackboneSpec:
    name = str(_get(cfg, "backbone", "tiny_cnn"))
    pretrained = bool(_get(cfg, "pretrained", False))
    if name in {"tiny", "tiny_cnn", "test_cnn"}:
        dim = int(_get(cfg, "embedding_dim", 64))
        return BackboneSpec(TinyBackbone(feature_dim=dim), dim)

    try:
        # On this Windows/conda setup, preloading Pillow avoids a torchvision DLL
        # load-order failure that can appear when timm imports torchvision first.
        from PIL import Image as _PILImage  # noqa: F401

        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for configured backbones. Install requirements.txt or use backbone=tiny_cnn for tests."
        ) from exc

    timm_kwargs = _as_plain_dict(_get(cfg, "timm_kwargs", {}))
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
        **timm_kwargs,
    )
    feature_dim = int(
        _get(cfg, "feature_dim", None)
        or getattr(model, "num_features", 0)
        or _get(cfg, "embedding_dim", 0)
    )
    if feature_dim <= 0:
        raise ValueError(f"Could not infer feature_dim for backbone {name}")
    return BackboneSpec(model, feature_dim)
