from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast_context(enabled: bool, device: torch.device):
    use_amp = enabled and device.type == "cuda"
    if not use_amp:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
