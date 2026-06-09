from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch


def get_git_commit(repo_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_info(repo_root: str | Path = ".") -> dict:
    packages = {}
    for name in [
        "torch",
        "timm",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
        "hydra-core",
        "omegaconf",
        "open-clip-torch",
        "huggingface_hub",
        "mamba-ssm",
        "causal-conv1d",
    ]:
        packages[name] = get_package_version(name)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": packages,
        "git_commit": get_git_commit(repo_root),
    }
