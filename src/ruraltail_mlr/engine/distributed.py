from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else default


def init_distributed(device_setting: str = "auto") -> DistributedState:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    enabled = world_size > 1

    use_cuda = torch.cuda.is_available() and device_setting != "cpu"
    if use_cuda:
        if enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        elif device_setting.startswith("cuda:"):
            device = torch.device(device_setting)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if enabled and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return DistributedState(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, state: DistributedState, src: int = 0):
    if not state.enabled:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def reduce_mean(value: float, state: DistributedState) -> float:
    if not state.enabled:
        return float(value)
    tensor = torch.tensor(float(value), device=state.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= state.world_size
    return float(tensor.item())


def per_device_batch_size(global_batch_size: int, state: DistributedState) -> int:
    if global_batch_size < 1:
        raise ValueError("train.batch_size must be >= 1")
    if not state.enabled:
        return int(global_batch_size)
    if global_batch_size % state.world_size != 0:
        raise ValueError(
            "For DDP, train.batch_size is treated as global batch size and must "
            f"be divisible by world_size={state.world_size}; got {global_batch_size}."
        )
    return int(global_batch_size // state.world_size)
