from pathlib import Path

import pytest
import torch

from ruraltail_mlr.losses.asl import AsymmetricLoss
from ruraltail_mlr.losses.bce import BCEWithLogitsLoss
from ruraltail_mlr.losses.decision import ASLWithTSoftF1Loss
from ruraltail_mlr.losses.factory import build_loss
from ruraltail_mlr.losses.focal import SigmoidFocalLoss


def test_formal_base_losses_are_finite_and_backward():
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    for loss_fn in [BCEWithLogitsLoss(), AsymmetricLoss(), SigmoidFocalLoss()]:
        logits = torch.tensor([[10.0, -10.0, 0.0], [-5.0, 5.0, 0.5]], requires_grad=True)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()


def test_asl_is_finite_for_extreme_half_precision_logits():
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)
    loss = AsymmetricLoss(eps=1e-8)(logits, targets)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_focal_is_finite_for_extreme_half_precision_logits():
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]], dtype=torch.float16, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)
    loss = SigmoidFocalLoss(gamma=2.0, alpha=0.25)(logits, targets)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_factory_builds_focal_loss():
    criterion = build_loss({"cls": {"name": "focal", "gamma": 2.0, "alpha": 0.25}})
    logits = torch.tensor([[0.2, -0.4]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    loss_pack = criterion(logits, targets)
    assert torch.isfinite(loss_pack["loss"])
    loss_pack["loss"].backward()


@pytest.mark.parametrize(
    "legacy_name",
    [
        "asl_t" + "_soft_f1",
        "t_aware" + "_asl",
        "decision_aware" + "_asl",
    ],
)
def test_talc_legacy_loss_aliases_are_not_public(legacy_name: str):
    with pytest.raises(ValueError, match="Unknown cls loss"):
        build_loss({"cls": {"name": legacy_name}})


def test_asl_soft_f1_t_loss_is_finite_and_backward(tmp_path: Path):
    freq = tmp_path / "class_frequency.csv"
    freq.write_text(
        "class_idx,class_name,train_count,train_freq,group\n"
        "0,head,100,0.50,head\n"
        "1,mid,20,0.10,medium\n"
        "2,tail,5,0.025,tail\n",
        encoding="utf-8",
    )
    logits = torch.tensor([[0.3, -0.2, 0.1], [-0.4, 0.6, -0.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss_pack = ASLWithTSoftF1Loss(freq)(logits, targets)

    assert torch.isfinite(loss_pack["loss"])
    assert "loss_asl" in loss_pack
    assert "loss_soft_f1" in loss_pack
    assert "loss_tail_recall" in loss_pack
    loss_pack["loss"].backward()
    assert logits.grad is not None
