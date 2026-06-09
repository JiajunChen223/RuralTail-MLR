from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import pandas as pd
import yaml


def first_present(mapping: dict, *keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def collect_results(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in run_dirs:
        cfg_path = run / "resolved_config.yaml"
        metrics_path = run / "metrics_test.json"
        hmt_path = run / "hmt_metrics_test.json"
        eff_path = run / "efficiency.json"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        hmt = json.loads(hmt_path.read_text(encoding="utf-8")) if hmt_path.exists() else {}
        eff = json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else {}
        rows.append(
            {
                "run_id": run.name,
                "seed": cfg.get("run", {}).get("seed"),
                "dataset": cfg.get("data", {}).get("name"),
                "model": cfg.get("model", {}).get("name"),
                "input_size": cfg.get("train", {}).get("input_size"),
                "backbone": cfg.get("model", {}).get("backbone"),
                "loss": cfg.get("loss", {}).get("cls", {}).get("name"),
                "threshold_strategy": cfg.get("eval", {}).get("threshold_strategy"),
                "threshold_tuning": cfg.get("eval", {}).get("tune_threshold_on_val"),
                "mAP": metrics.get("mAP"),
                "Macro_F1": first_present(metrics, "macro_F1", "mCF1"),
                "Micro_F1": metrics.get("micro_F1"),
                "Tail_mAP": first_present(hmt, "Tail_mAP", "tail_mAP"),
                "Tail_F1": first_present(hmt, "Tail_F1", "tail_F1"),
                "Params": eff.get("params"),
                "FLOPs": eff.get("flops"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run metrics into a CSV table.")
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_dirs = sorted([p for p in Path(args.runs).glob("*") if p.is_dir()])
    df = collect_results(run_dirs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
