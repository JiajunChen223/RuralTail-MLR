from types import SimpleNamespace

import torch

from ruraltail_mlr.data.label_schema import default_class_mapping, write_class_mapping
from ruraltail_mlr.models.factory import build_model


def test_linear_forward_shape(tmp_path):
    mapping = tmp_path / "class_mapping.json"
    write_class_mapping(default_class_mapping(), mapping)
    x = torch.randn(2, 3, 32, 32)
    linear_cfg = SimpleNamespace(backbone="tiny_cnn", head="linear", embedding_dim=32, dropout=0.0)
    linear = build_model(linear_cfg, str(mapping))
    assert linear(x).shape == (2, 18)
