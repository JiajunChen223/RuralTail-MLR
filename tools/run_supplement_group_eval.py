from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader

from ruraltail_mlr.data.dataset import ChinaMASDataset
from ruraltail_mlr.data.label_schema import class_names_from_mapping, load_class_mapping, mapping_hash
from ruraltail_mlr.data.transforms import build_transform
from ruraltail_mlr.engine.checkpoint import load_checkpoint, validate_checkpoint_metadata
from ruraltail_mlr.engine.evaluator import Evaluator
from ruraltail_mlr.metrics.threshold import apply_thresholds, tune_thresholds_on_val
from ruraltail_mlr.models.factory import build_model
from ruraltail_mlr.utils.io import save_json


ROOT = Path(__file__).resolve().parents[1]
FORMAL_GRID = [round(0.10 + 0.05 * idx, 2) for idx in range(13)]
SCORE_SOURCE_LABELS = {
    "bce": "BCE",
    "focal": "Focal",
    "asl": "ASL",
    "talc": "TALC",
}
METHOD_ORDER = list(SCORE_SOURCE_LABELS)
FORMAL_OPERATING_RULES = {
    "fixed": {"suffix": "F", "threshold": 0.5},
    "group": {"suffix": "G"},
}
DIAGNOSTIC_OPERATING_RULES = ["global", "classwise"]


def display_method(method: str, operating_rule: str) -> str:
    suffix = FORMAL_OPERATING_RULES[operating_rule]["suffix"]
    return f"{SCORE_SOURCE_LABELS.get(method, method)}-{suffix}"


def threshold_triplet(threshold, class_names: list[str], hmt: dict) -> dict[str, float]:
    if isinstance(threshold, dict) and isinstance(threshold.get("thresholds"), dict):
        values = threshold["thresholds"]
        default = float(values.get("default", threshold.get("threshold", 0.5)))
        return {
            "threshold_H": float(values.get("head", default)),
            "threshold_M": float(values.get("medium", default)),
            "threshold_T": float(values.get("tail", default)),
        }
    if isinstance(threshold, dict) and isinstance(threshold.get("thresholds"), list):
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        values = threshold["thresholds"]
        out = {}
        for key, group in [("threshold_H", "head"), ("threshold_M", "medium"), ("threshold_T", "tail")]:
            idxs = [name_to_idx[name] for name in hmt.get(group, []) if name in name_to_idx]
            out[key] = float(sum(values[idx] for idx in idxs) / max(len(idxs), 1))
        return out
    scalar = float(threshold if not isinstance(threshold, dict) else threshold.get("threshold", 0.5))
    return {"threshold_H": scalar, "threshold_M": scalar, "threshold_T": scalar}


def load_records(manifest_path: Path, datasets: set[str], methods: set[str]) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record
        for record in payload["records"]
        if record["dataset"] in datasets and record["method"] in methods
    ]
    return sorted(records, key=lambda item: (item["dataset"], item["model"], item["method"]))


def resolve_package_path(relative_path: str) -> Path:
    return ROOT / Path(relative_path)


def load_dataset_cfg(dataset: str):
    return OmegaConf.load(ROOT / "configs" / "data" / f"{dataset}.yaml")


def make_loader(cfg, split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = ChinaMASDataset(
        images_index_csv=cfg.data.images_index,
        labels_csv=cfg.data.labels,
        split_csv=cfg.data.split,
        split=split,
        class_mapping_json=cfg.data.class_mapping,
        transform=build_transform(cfg.train.input_size, train=False),
        image_root=cfg.data.image_root,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def tail_operating_metrics(
    y_true, y_prob, threshold, class_names: list[str], hmt: dict
) -> dict[str, float]:
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idxs = [name_to_idx[name] for name in hmt.get("tail", []) if name in name_to_idx]
    if not idxs:
        return {"tail_precision": float("nan"), "tail_recall": float("nan")}
    y_pred = apply_thresholds(y_prob, threshold, class_names, hmt)
    return {
        "tail_precision": float(
            precision_score(y_true[:, idxs], y_pred[:, idxs], average="macro", zero_division=0)
        ),
        "tail_recall": float(
            recall_score(y_true[:, idxs], y_pred[:, idxs], average="macro", zero_division=0)
        ),
    }


def enrich_metrics(
    metrics: dict,
    hmt_metrics: dict,
    y_true,
    y_prob,
    threshold,
    class_names: list[str],
    hmt: dict,
) -> dict:
    combined = dict(metrics)
    combined.update(hmt_metrics)
    combined.update(tail_operating_metrics(y_true, y_prob, threshold, class_names, hmt))
    return combined


def write_predictions(predictions: pd.DataFrame, split: str, out_path: Path) -> None:
    predictions = predictions.copy()
    predictions.insert(1, "split", split)
    predictions.to_csv(out_path, index=False)


def selected_tail_threshold(spec: dict) -> float:
    thresholds = spec.get("thresholds", {})
    if isinstance(thresholds, dict):
        return float(thresholds["tail"])
    raise ValueError(f"Expected group threshold specification, got: {spec}")


def build_tail_response(
    evaluator: Evaluator,
    val_logits: torch.Tensor,
    val_targets: torch.Tensor,
    val_ids: list[str],
    selected_spec: dict,
    class_names: list[str],
    hmt: dict,
) -> pd.DataFrame:
    y_true = val_targets.numpy().astype(int)
    y_prob = torch.sigmoid(val_logits).numpy()
    rows = []
    for tail_threshold in FORMAL_GRID:
        threshold = deepcopy(selected_spec)
        threshold["thresholds"]["tail"] = float(tail_threshold)
        metrics, _, hmt_metrics, _ = evaluator.evaluate(
            val_logits, val_targets, val_ids, threshold=threshold
        )
        metrics = enrich_metrics(
            metrics, hmt_metrics, y_true, y_prob, threshold, class_names, hmt
        )
        rows.append(
            {
                "tail_threshold": tail_threshold,
                "macro_F1": metrics["macro_F1"],
                "tail_F1": metrics["tail_F1"],
                "tail_precision": metrics["tail_precision"],
                "tail_recall": metrics["tail_recall"],
                "selected_for_run": tail_threshold == selected_tail_threshold(selected_spec),
            }
        )
    return pd.DataFrame(rows)


def evaluate_record(record: dict, args: argparse.Namespace) -> None:
    dataset = record["dataset"]
    model_name = record["model"]
    method = record["method"]
    out_dir = Path(args.output_root) / dataset / model_name / method
    done = out_dir / "DONE"
    if done.exists() and not args.overwrite:
        print(f"SKIP {dataset}/{model_name}/{method}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(resolve_package_path(record["config"]))
    OmegaConf.set_struct(cfg, False)
    cfg.data = load_dataset_cfg(dataset)
    cfg.model.pretrained = False

    mapping = load_class_mapping(cfg.data.class_mapping)
    class_names = class_names_from_mapping(mapping)
    hmt = json.loads(Path(cfg.data.hmt).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model, cfg.data.class_mapping)
    checkpoint_path = resolve_package_path(record["checkpoint"])
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    try:
        validate_checkpoint_metadata(checkpoint, class_mapping_hash=mapping_hash(mapping))
    except ValueError as exc:
        if not args.allow_legacy_metadata_mismatch:
            raise
        print(f"WARN {dataset}/{model_name}/{method}: {exc}", flush=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    batch_size = int(args.batch_size or cfg.train.batch_size)
    evaluator = Evaluator(class_names, hmt, device)
    val_loader = make_loader(cfg, "val", batch_size, args.num_workers)
    test_loader = make_loader(cfg, "test", batch_size, args.num_workers)
    print(f"PREDICT VAL  {dataset}/{model_name}/{method}", flush=True)
    val_logits, val_targets, val_ids = evaluator.predict(model, val_loader)
    print(f"PREDICT TEST {dataset}/{model_name}/{method}", flush=True)
    test_logits, test_targets, test_ids = evaluator.predict(model, test_loader)

    val_probs = torch.sigmoid(val_logits).numpy()
    val_true = val_targets.numpy().astype(int)
    fixed_threshold = 0.5
    group_threshold, best_score = tune_thresholds_on_val(
        val_true,
        val_probs,
        strategy="group",
        class_names=class_names,
        class_groups=hmt,
        grid=FORMAL_GRID,
        fallback_threshold=0.5,
    )
    save_json(
        {
            "protocol": "fixed 0.5 operating rule",
            "threshold_spec": fixed_threshold,
        },
        out_dir / "threshold_fixed.json",
    )
    save_json(
        {
            "protocol": "validation-only group-threshold selection",
            "grid": FORMAL_GRID,
            "grid_step": 0.05,
            "objective": "group macro-F1",
            "validation_macro_F1": float(best_score),
            "threshold_spec": group_threshold,
        },
        out_dir / "threshold_group_val.json",
    )

    formal_thresholds = {
        "fixed": fixed_threshold,
        "group": group_threshold,
    }
    for operating_rule, threshold in formal_thresholds.items():
        for split, logits, targets, ids in [
            ("val", val_logits, val_targets, val_ids),
            ("test", test_logits, test_targets, test_ids),
        ]:
            metrics, per_class, hmt_metrics, predictions = evaluator.evaluate(
                logits, targets, ids, threshold=threshold
            )
            y_true = targets.numpy().astype(int)
            y_prob = torch.sigmoid(logits).numpy()
            enriched = enrich_metrics(
                metrics, hmt_metrics, y_true, y_prob, threshold, class_names, hmt
            )
            enriched["operating_rule"] = operating_rule
            enriched["display_method"] = display_method(method, operating_rule)
            save_json(enriched, out_dir / f"metrics_{split}_{operating_rule}.json")
            pd.DataFrame(per_class).to_csv(
                out_dir / f"per_class_{split}_{operating_rule}.csv", index=False
            )
            write_predictions(
                predictions, split, out_dir / f"predictions_{split}_{operating_rule}.csv"
            )

    diagnostic_thresholds = {}
    diagnostic_meta = {}
    for strategy in DIAGNOSTIC_OPERATING_RULES:
        diagnostic_threshold, diagnostic_score = tune_thresholds_on_val(
            val_true,
            val_probs,
            strategy=strategy,
            class_names=class_names,
            class_groups=hmt,
            grid=FORMAL_GRID,
            fallback_threshold=0.5,
        )
        diagnostic_thresholds[strategy] = diagnostic_threshold
        diagnostic_meta[strategy] = {
            "validation_macro_F1": float(diagnostic_score),
            "threshold_spec": diagnostic_threshold,
        }
    save_json(
        {
            "protocol": "validation-only diagnostic threshold selection",
            "grid": FORMAL_GRID,
            "grid_step": 0.05,
            "objective": "macro-F1",
            "diagnostic_only": True,
            "rules": diagnostic_meta,
        },
        out_dir / "threshold_diagnostic_val.json",
    )
    for strategy, threshold in diagnostic_thresholds.items():
        metrics, _, hmt_metrics, _ = evaluator.evaluate(
            test_logits, test_targets, test_ids, threshold=threshold
        )
        y_true = test_targets.numpy().astype(int)
        y_prob = torch.sigmoid(test_logits).numpy()
        enriched = enrich_metrics(metrics, hmt_metrics, y_true, y_prob, threshold, class_names, hmt)
        enriched["operating_rule"] = strategy
        enriched["diagnostic_only"] = True
        save_json(enriched, out_dir / f"metrics_test_diagnostic_{strategy}.json")

    if dataset == "china_mas_50k":
        response = build_tail_response(
            evaluator, val_logits, val_targets, val_ids, group_threshold, class_names, hmt
        )
        response.to_csv(out_dir / "tail_threshold_response_val.csv", index=False)

    save_json(record, out_dir / "run_spec.json")
    done.write_text("ok\n", encoding="utf-8")
    print(f"DONE {dataset}/{model_name}/{method}", flush=True)


def collect_test_summary(output_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in output_root.glob("*/*/*/metrics_test_*.json"):
        if "diagnostic" in metrics_path.name:
            continue
        dataset, model, method = metrics_path.relative_to(output_root).parts[:3]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        operating_rule = metrics_path.stem.replace("metrics_test_", "")
        if operating_rule not in FORMAL_OPERATING_RULES:
            continue
        if operating_rule == "fixed":
            threshold = json.loads(
                (metrics_path.parent / "threshold_fixed.json").read_text(encoding="utf-8")
            )["threshold_spec"]
        else:
            threshold = json.loads(
                (metrics_path.parent / "threshold_group_val.json").read_text(encoding="utf-8")
            )["threshold_spec"]
        thresholds = threshold_triplet(threshold, [], {}) if operating_rule == "fixed" else threshold_triplet(threshold, [], {})
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "score_source": method,
                "score_source_display": SCORE_SOURCE_LABELS.get(method, method),
                "operating_rule": operating_rule,
                "display_method": display_method(method, operating_rule),
                "mAP": metrics["mAP"],
                "macro_F1": metrics["macro_F1"],
                "tail_mAP": metrics["tail_mAP"],
                "tail_precision": metrics["tail_precision"],
                "tail_recall": metrics["tail_recall"],
                "tail_F1": metrics["tail_F1"],
                **thresholds,
            }
        )
    return pd.DataFrame(rows)


def collect_diagnostic_summary(output_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in output_root.glob("*/*/*/metrics_test_diagnostic_*.json"):
        dataset, model, method = metrics_path.relative_to(output_root).parts[:3]
        operating_rule = metrics_path.stem.replace("metrics_test_diagnostic_", "")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "score_source": method,
                "score_source_display": SCORE_SOURCE_LABELS.get(method, method),
                "operating_rule": operating_rule,
                "diagnostic_only": True,
                "mAP": metrics["mAP"],
                "macro_F1": metrics["macro_F1"],
                "tail_mAP": metrics["tail_mAP"],
                "tail_precision": metrics["tail_precision"],
                "tail_recall": metrics["tail_recall"],
                "tail_F1": metrics["tail_F1"],
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_response(response_mean: pd.DataFrame, out_root: Path) -> None:
    china = response_mean.loc[response_mean["dataset"] == "china_mas_50k"].copy()
    if china.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), dpi=240)
    colors = {"bce": "#6c8ebf", "focal": "#c66b46", "asl": "#7c5ba7", "talc": "#148547"}
    for method in METHOD_ORDER:
        values = china.loc[china["method"] == method].sort_values("tail_threshold")
        if values.empty:
            continue
        axes[0].plot(
            values["tail_threshold"],
            values["tail_F1"],
            marker="o",
            linewidth=1.4,
            markersize=3.0,
            label=f"{SCORE_SOURCE_LABELS[method]}-G",
            color=colors[method],
        )
    axes[0].set_title("(a) Operating-boundary score learners", fontsize=9)
    axes[0].set_xlabel(r"Tail threshold $\tau_T$")
    axes[0].set_ylabel("Validation Tail F1")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    talc = china.loc[china["method"] == "talc"].sort_values("tail_threshold")
    for metric, label, color in [
        ("tail_precision", "Precision", "#1c5aa6"),
        ("tail_recall", "Recall", "#bf4b47"),
        ("tail_F1", "F1", "#148547"),
    ]:
        axes[1].plot(
            talc["tail_threshold"],
            talc[metric],
            marker="o",
            linewidth=1.4,
            markersize=3.0,
            label=label,
            color=color,
        )
    axes[1].set_title("(b) TALC operating response", fontsize=9)
    axes[1].set_xlabel(r"Tail threshold $\tau_T$")
    axes[1].set_ylabel("Validation score")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.5)
        ax.set_xlim(0.09, 0.71)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_root / "threshold_response_validation.pdf", bbox_inches="tight")
    fig.savefig(out_root / "threshold_response_validation.png", bbox_inches="tight")
    plt.close(fig)


def summarize(output_root: Path) -> None:
    summary = collect_test_summary(output_root)
    if summary.empty:
        raise FileNotFoundError(f"No completed supplement results under {output_root}")
    summary.to_csv(output_root / "test_results_by_carrier.csv", index=False)
    averages = (
        summary.groupby(
            [
                "dataset",
                "method",
                "score_source",
                "score_source_display",
                "operating_rule",
                "display_method",
            ],
            as_index=False,
        )[
            ["mAP", "macro_F1", "tail_mAP", "tail_precision", "tail_recall", "tail_F1"]
        ]
        .mean()
        .sort_values(["dataset", "method", "operating_rule"])
    )
    averages.to_csv(output_root / "test_results_method_average.csv", index=False)
    diagnostics = collect_diagnostic_summary(output_root)
    if not diagnostics.empty:
        diagnostics.to_csv(output_root / "diagnostic_results_by_carrier.csv", index=False)
        (
            diagnostics.groupby(
                [
                    "dataset",
                    "method",
                    "score_source",
                    "score_source_display",
                    "operating_rule",
                    "diagnostic_only",
                ],
                as_index=False,
            )[["mAP", "macro_F1", "tail_mAP", "tail_precision", "tail_recall", "tail_F1"]]
            .mean()
            .sort_values(["dataset", "method", "operating_rule"])
            .to_csv(output_root / "diagnostic_results_method_average.csv", index=False)
        )

    response_files = list(output_root.glob("china_mas_50k/*/*/tail_threshold_response_val.csv"))
    if response_files:
        response_rows = []
        for path in response_files:
            _, model, method = path.relative_to(output_root).parts[:3]
            frame = pd.read_csv(path)
            frame.insert(0, "dataset", "china_mas_50k")
            frame.insert(1, "model", model)
            frame.insert(2, "method", method)
            response_rows.append(frame)
        response = pd.concat(response_rows, ignore_index=True)
        response.to_csv(output_root / "threshold_response_validation_by_carrier.csv", index=False)
        response_mean = (
            response.groupby(["dataset", "method", "tail_threshold"], as_index=False)[
                ["macro_F1", "tail_F1", "tail_precision", "tail_recall"]
            ]
            .mean()
            .sort_values(["method", "tail_threshold"])
        )
        response_mean.to_csv(output_root / "threshold_response_validation_mean.csv", index=False)
        plot_threshold_response(response_mean, output_root)
    print(f"WROTE summaries under {output_root}", flush=True)


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(
        description="Run formal fixed/group paper-protocol operating-rule evaluation."
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/eval_manifest_seed2026.json",
        help="Manifest path relative to the package root.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/supplement_group_eval_seed2026",
        help="Output directory relative to the package root.",
    )
    parser.add_argument("--datasets", default="china_mas_50k,agriculture_vision_2021")
    parser.add_argument("--methods", default="bce,focal,asl,talc")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-legacy-metadata-mismatch", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output_root = str(ROOT / args.output_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(output_root)
        return

    records = load_records(
        ROOT / args.manifest,
        set(args.datasets.split(",")),
        set(args.methods.split(",")),
    )
    shard = [record for idx, record in enumerate(records) if idx % args.num_shards == args.shard_index]
    print(f"FORMAL_GRID={FORMAL_GRID}", flush=True)
    print(f"TOTAL={len(records)} SHARD={args.shard_index}/{args.num_shards} N={len(shard)}", flush=True)
    for record in shard:
        evaluate_record(record, args)
    if args.num_shards == 1:
        summarize(output_root)


if __name__ == "__main__":
    main()
