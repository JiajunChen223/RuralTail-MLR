from ruraltail_mlr.data.label_schema import (
    DEFAULT_CLASS_NAMES,
    class_names_from_mapping,
    default_class_mapping,
    validate_label_columns,
)


def test_default_class_mapping_locks_china_mas_order():
    mapping = default_class_mapping()
    assert mapping["num_classes"] == 18
    assert class_names_from_mapping(mapping) == DEFAULT_CLASS_NAMES


def test_validate_label_columns_requires_all_classes():
    validate_label_columns(["image_id", *DEFAULT_CLASS_NAMES], DEFAULT_CLASS_NAMES)
    try:
        validate_label_columns(["image_id", *DEFAULT_CLASS_NAMES[:-1]], DEFAULT_CLASS_NAMES)
    except ValueError as exc:
        assert "Missing label columns" in str(exc)
    else:
        raise AssertionError("validate_label_columns should fail on missing labels")
