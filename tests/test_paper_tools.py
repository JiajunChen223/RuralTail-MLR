import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    tools_dir = ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_eval_manifest_collects_public_score_sources(tmp_path: Path):
    module = _load_tool("build_eval_manifest")
    runs = tmp_path / "runs"
    for run_name, loss_name in [
        ("r_bce", "bce"),
        ("r_focal", "focal"),
        ("r_asl", "asl"),
        ("r_talc", "asl_soft_f1"),
    ]:
        run = runs / run_name
        run.mkdir(parents=True)
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "run": {"name": run_name},
                    "data": {"name": "china_mas_50k"},
                    "model": {"name": "resnet50_linear", "backbone": "resnet50"},
                    "loss": {"cls": {"name": loss_name}},
                }
            ),
            encoding="utf-8",
        )
        torch.save({"model_state": {}}, run / "checkpoint_best.pth")
    legacy = runs / "r_legacy_talc_alias"
    legacy.mkdir(parents=True)
    legacy_name = "t_aware" + "_asl"
    (legacy / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"name": "r_legacy_talc_alias"},
                "data": {"name": "china_mas_50k"},
                "model": {"name": "resnet50_linear", "backbone": "resnet50"},
                "loss": {"cls": {"name": legacy_name}},
            }
        ),
        encoding="utf-8",
    )
    torch.save({"model_state": {}}, legacy / "checkpoint_best.pth")

    records = module.collect_records(runs, {"bce", "focal", "asl", "talc"}, {"china_mas_50k"})
    assert [record["method"] for record in records] == ["asl", "bce", "focal", "talc"]
    assert [record["score_source_display"] for record in records] == ["ASL", "BCE", "Focal", "TALC"]
    assert all("display_method" not in record for record in records)


def test_make_paper_tables_writes_paper_columns(tmp_path: Path):
    module = _load_tool("make_paper_tables")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    frame = pd.DataFrame(
        [
            {
                "dataset": "china_mas_50k",
                "model": "resnet50",
                "method": "talc",
                "score_source": "talc",
                "score_source_display": "TALC",
                "operating_rule": "group",
                "display_method": "TALC-G",
                "mAP": 0.8,
                "macro_F1": 0.7,
                "tail_mAP": 0.6,
                "tail_precision": 0.5,
                "tail_recall": 0.4,
                "tail_F1": 0.45,
            }
        ]
    )
    frame.to_csv(eval_root / "test_results_by_carrier.csv", index=False)
    frame.to_csv(eval_root / "test_results_method_average.csv", index=False)

    out = tmp_path / "tables"
    module.write_tables(eval_root, out)
    result = pd.read_csv(out / "paper_results_by_carrier.csv")
    assert {"Macro-F1", "Tail mAP", "Tail P", "Tail R", "Tail F1"}.issubset(result.columns)
    blocked = "|".join(["B" + "OR", "DAT", "classwise", "global"])
    assert not result["display_method"].astype(str).str.contains(blocked).any()


def test_make_paper_tables_rejects_nonformal_main_rows(tmp_path: Path):
    module = _load_tool("make_paper_tables")
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    frame = pd.DataFrame(
        [
            {
                "dataset": "china_mas_50k",
                "model": "resnet50",
                "method": "asl",
                "operating_rule": "group",
                "display_method": "ASL-G / " + "B" + "OR",
                "mAP": 0.8,
                "macro_F1": 0.7,
                "tail_mAP": 0.6,
                "tail_precision": 0.5,
                "tail_recall": 0.4,
                "tail_F1": 0.45,
            }
        ]
    )
    frame.to_csv(eval_root / "test_results_by_carrier.csv", index=False)
    frame.to_csv(eval_root / "test_results_method_average.csv", index=False)

    try:
        module.write_tables(eval_root, tmp_path / "tables")
    except ValueError as exc:
        assert "non-formal" in str(exc)
    else:
        raise AssertionError("analysis alias display must be rejected from paper-facing tables")


def test_collect_summary_expands_fixed_and_group_rows(tmp_path: Path):
    module = _load_tool("run_supplement_group_eval")
    run = tmp_path / "eval" / "china_mas_50k" / "resnet50" / "asl"
    run.mkdir(parents=True)
    (run / "threshold_fixed.json").write_text(
        json.dumps({"threshold_spec": 0.5}), encoding="utf-8"
    )
    (run / "threshold_group_val.json").write_text(
        json.dumps(
            {"threshold_spec": {"strategy": "group", "thresholds": {"head": 0.6, "medium": 0.5, "tail": 0.4}}}
        ),
        encoding="utf-8",
    )
    metrics = {
        "mAP": 0.8,
        "macro_F1": 0.7,
        "tail_mAP": 0.6,
        "tail_precision": 0.5,
        "tail_recall": 0.4,
        "tail_F1": 0.45,
    }
    (run / "metrics_test_fixed.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "metrics_test_group.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "metrics_test_diagnostic_global.json").write_text(json.dumps(metrics), encoding="utf-8")

    summary = module.collect_test_summary(tmp_path / "eval")
    assert set(summary["display_method"]) == {"ASL-F", "ASL-G"}
    assert set(summary["operating_rule"]) == {"fixed", "group"}
    assert not summary["display_method"].astype(str).str.contains("B" + "OR").any()
    diagnostics = module.collect_diagnostic_summary(tmp_path / "eval")
    assert diagnostics["operating_rule"].tolist() == ["global"]


def test_manifest_example_uses_placeholder_not_large_checkpoint():
    payload = json.loads((ROOT / "supplement" / "seed2026_group_eval_manifest.example.json").read_text(encoding="utf-8"))
    assert payload["record_count"] == 1
    assert payload["records"][0]["checkpoint_bytes"] == 0
