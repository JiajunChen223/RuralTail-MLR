from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a simple failure-case CSV for manual figure assembly.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="artifacts/reports/failure_cases.csv")
    args = parser.parse_args()
    preds = pd.read_csv(args.predictions)
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    true_cols = [c for c in preds.columns if c.startswith("y_true_")]
    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    rows = []
    for _, row in preds.iterrows():
        fp = sum(int(row[p]) == 1 and int(row[t]) == 0 for p, t in zip(pred_cols, true_cols))
        fn = sum(int(row[p]) == 0 and int(row[t]) == 1 for p, t in zip(pred_cols, true_cols))
        rows.append({"image_id": row["image_id"], "false_positive": fp, "false_negative": fn, "max_prob": max(row[c] for c in prob_cols)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["false_negative", "false_positive"], ascending=False).head(64).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
