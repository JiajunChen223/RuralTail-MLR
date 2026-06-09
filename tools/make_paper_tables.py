from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_COLUMNS = {
    "macro_F1": "Macro-F1",
    "tail_mAP": "Tail mAP",
    "tail_precision": "Tail P",
    "tail_recall": "Tail R",
    "tail_F1": "Tail F1",
}
MODEL_ORDER_CHINA = [
    "recent_mlmamba_resnet18",
    "recent_sfin_resnet18",
    "resnet50",
    "mambaout_s",
    "efficientnetv2_s",
    "pvtv2_b2",
]
MODEL_ORDER_AGV = ["resnet50", "pvtv2_b2", "mambaout_s"]
FORMAL_OPERATING_RULES = {"fixed", "group"}
FORBIDDEN_MAIN_TABLE_TOKENS = ("B" + "OR", "DAT", "classwise", "global")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=PAPER_COLUMNS)


def _formal_only(df: pd.DataFrame) -> pd.DataFrame:
    if "operating_rule" in df.columns:
        df = df.loc[df["operating_rule"].isin(FORMAL_OPERATING_RULES)].copy()
    if "diagnostic_only" in df.columns:
        df = df.loc[~df["diagnostic_only"].astype(bool)].copy()
    text = "\n".join(str(value) for value in df.astype(str).to_numpy().ravel())
    found = [token for token in FORBIDDEN_MAIN_TABLE_TOKENS if token in text]
    if found:
        raise ValueError(f"Paper-facing tables contain non-formal tokens: {found}")
    return df


def write_tables(eval_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_carrier = _rename_metrics(_formal_only(_read(eval_root / "test_results_by_carrier.csv")))
    averages = _rename_metrics(_formal_only(_read(eval_root / "test_results_method_average.csv")))
    by_carrier.to_csv(out_dir / "paper_results_by_carrier.csv", index=False)
    averages.to_csv(out_dir / "paper_results_method_average.csv", index=False)

    china = by_carrier.loc[by_carrier["dataset"] == "china_mas_50k"].copy()
    china["model"] = pd.Categorical(china["model"], MODEL_ORDER_CHINA, ordered=True)
    china.sort_values(["display_method", "model"]).to_csv(
        out_dir / "table_china_carrier_tail.csv",
        index=False,
    )

    agv = by_carrier.loc[by_carrier["dataset"] == "agriculture_vision_2021"].copy()
    agv["model"] = pd.Categorical(agv["model"], MODEL_ORDER_AGV, ordered=True)
    agv.sort_values(["display_method", "model"]).to_csv(
        out_dir / "table_agv_carrier_tail.csv",
        index=False,
    )

    response = eval_root / "threshold_response_validation_mean.csv"
    if response.exists():
        _read(response).to_csv(out_dir / "threshold_response_validation_mean.csv", index=False)

    diagnostic = eval_root / "diagnostic_results_by_carrier.csv"
    if diagnostic.exists():
        _read(diagnostic).to_csv(out_dir / "diagnostic_results_by_carrier.csv", index=False)

    print(f"wrote paper tables under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-facing CSV tables from paper-protocol evaluation outputs.")
    parser.add_argument("--eval-root", default="artifacts/supplement_group_eval_seed2026")
    parser.add_argument("--out-dir", default="artifacts/paper_tables")
    args = parser.parse_args()
    write_tables(ROOT / args.eval_root, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
