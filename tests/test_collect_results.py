import json
import importlib.util
import sys
from pathlib import Path

import yaml


def _load_collect_results():
    root = Path(__file__).resolve().parents[1]
    tools_dir = root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "collect_results.py"
    spec = importlib.util.spec_from_file_location("collect_results_tool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.collect_results


def test_collect_results_preserves_zero_tail_map(tmp_path: Path):
    collect_results = _load_collect_results()
    run = tmp_path / "run"
    run.mkdir()
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"seed": 1},
                "data": {"name": "china_mas_50k"},
                "train": {"input_size": 224},
                "model": {"name": "tiny_linear", "backbone": "tiny"},
                "eval": {"threshold_strategy": "global", "tune_threshold_on_val": False},
                "loss": {"cls": {"name": "asl"}},
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics_test.json").write_text(
        json.dumps({"mAP": 0.1, "macro_F1": 0.2, "micro_F1": 0.3}),
        encoding="utf-8",
    )
    (run / "hmt_metrics_test.json").write_text(json.dumps({"Tail_mAP": 0.0, "Tail_F1": 0.0}), encoding="utf-8")
    (run / "efficiency.json").write_text(json.dumps({"params": 10, "flops": None}), encoding="utf-8")
    df = collect_results([run])
    assert df.loc[0, "Tail_mAP"] == 0.0
