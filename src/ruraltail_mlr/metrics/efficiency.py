from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def measure_inference_time(model: nn.Module, input_size: int, device: torch.device, repeats: int = 20) -> float:
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return float((time.perf_counter() - start) / repeats)
