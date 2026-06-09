import pytest

from ruraltail_mlr.engine.checkpoint import validate_checkpoint_metadata


def test_checkpoint_metadata_validation_rejects_mismatch():
    ckpt = {"class_mapping_hash": "abc"}
    validate_checkpoint_metadata(ckpt, class_mapping_hash="abc")
    with pytest.raises(ValueError):
        validate_checkpoint_metadata(ckpt, class_mapping_hash="different")
