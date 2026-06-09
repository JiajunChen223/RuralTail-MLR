import pytest
import torch

from ruraltail_mlr.engine.distributed import DistributedState, per_device_batch_size


def test_per_device_batch_size_single_process():
    state = DistributedState(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert per_device_batch_size(64, state) == 64


def test_per_device_batch_size_ddp_uses_global_batch_size():
    state = DistributedState(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )
    assert per_device_batch_size(64, state) == 32


def test_per_device_batch_size_ddp_rejects_non_divisible_global_batch():
    state = DistributedState(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError):
        per_device_batch_size(65, state)
