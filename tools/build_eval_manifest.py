from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCORE_SOURCE_LABELS = {
    "bce": "BCE",
    "focal": "Focal",
    "asl": "ASL",
    "talc": "TALC",
}
MODEL_ALIASES = {
    "resnet50_linear": "resnet50",
    "efficientnetv2_s_linear": "efficientnetv2_s",
    "pvtv2_b2_linear": "pvtv2_b2",
    "mambaout_s_linear": "mambaout_s",
    "sfin": "recent_sfin_resnet18",
    "mlmamba": "recent_mlmamba_resnet18",
}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _method_from_config(cfg: dict) -> str | None:
    loss_name = str(cfg.get("loss", {}).get("cls", {}).get("name", "")).lower()
    run_name = str(cfg.get("run", {}).get("name", "")).lower()
    if loss_name == "bce":
        return "bce"
    if loss_name in {"focal", "sigmoid_focal"}:
        return "focal"
    if loss_name == "asl_soft_f1":
        return "talc"
    if loss_name == "asl" and "asl_softf1" not in run_name:
        return "asl"
    return None


def _model_from_config(cfg: dict) -> str:
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", ""))
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    backbone = str(model_cfg.get("backbone", ""))
    if backbone == "resnet50":
        return "resnet50"
    if backbone == "tf_efficientnetv2_s":
        return "efficientnetv2_s"
    if backbone == "pvt_v2_b2":
        return "pvtv2_b2"
    if backbone == "mambaout_small":
        return "mambaout_s"
    return model_name or backbone


def collect_records(runs_dir: Path, methods: set[str], datasets: set[str]) -> list[dict]:
    records = []
    for run_dir in sorted(path for path in runs_dir.glob("*") if path.is_dir()):
        cfg_path = run_dir / "resolved_config.yaml"
        ckpt_path = run_dir / "checkpoint_best.pth"
        if not cfg_path.exists() or not ckpt_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        dataset = str(cfg.get("data", {}).get("name", ""))
        method = _method_from_config(cfg)
        if method is None or method not in methods or dataset not in datasets:
            continue
        model = _model_from_config(cfg)
        records.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "score_source": method,
                "score_source_display": SCORE_SOURCE_LABELS[method],
                "source_run_name": run_dir.name,
                "checkpoint": _rel(ckpt_path),
                "config": _rel(cfg_path),
                "checkpoint_bytes": ckpt_path.stat().st_size,
                "checkpoint_sha256": _sha256(ckpt_path),
            }
        )
    return sorted(records, key=lambda item: (item["dataset"], item["model"], item["method"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paper-protocol evaluation manifest from completed runs.")
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--out", default="artifacts/eval_manifest_seed2026.json")
    parser.add_argument("--datasets", default="china_mas_50k,agriculture_vision_2021")
    parser.add_argument("--methods", default="bce,focal,asl,talc")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    runs_dir = ROOT / args.runs
    records = collect_records(
        runs_dir,
        methods={value.strip() for value in args.methods.split(",") if value.strip()},
        datasets={value.strip() for value in args.datasets.split(",") if value.strip()},
    )
    payload = {
        "package": "RuralTail-MLR paper protocol generated manifest",
        "seed": args.seed,
        "purpose": "Fixed and validation-selected H/M/T operating-rule evaluation for paper tables.",
        "formal_protocol": {
            "score_sources": ["bce", "focal", "asl", "talc"],
            "formal_operating_rules": ["fixed", "group"],
            "fixed_threshold": 0.5,
            "group_strategy": "validation-selected H/M/T group thresholds",
            "group_objective": "group macro-F1",
            "group_grid": [round(0.10 + 0.05 * idx, 2) for idx in range(13)],
            "group_step": 0.05,
            "diagnostic_operating_rules": ["global", "classwise"],
            "test_time_adaptation": False,
        },
        "raw_datasets_included": False,
        "record_count": len(records),
        "records": records,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} records={len(records)}")


if __name__ == "__main__":
    main()
