from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Head/Medium/Tail metrics for latest run.")
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--out", default="artifacts/figures/hmt_map_bar.png")
    args = parser.parse_args()
    latest = sorted(Path(args.runs).glob("*/hmt_metrics_test.json"))[-1]
    hmt = json.loads(latest.read_text(encoding="utf-8"))
    labels = ["head", "medium", "tail"]
    values = [hmt.get(f"{x}_mAP", 0.0) for x in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values)
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
