from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-class AP sorted by train frequency.")
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--frequency", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    latest = sorted(Path(args.runs).glob("*/per_class_ap_test.csv"))[-1]
    ap = pd.read_csv(latest)
    freq = pd.read_csv(args.frequency)
    df = ap.merge(freq, on=["class_idx", "class_name"], how="left").sort_values("rank")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["class_name"], df["AP"])
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
