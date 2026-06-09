from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_container(cfg: Any) -> dict:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def save_resolved_config(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, DictConfig):
        path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    else:
        import yaml

        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
