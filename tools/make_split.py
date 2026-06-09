from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from ruraltail_mlr.data.split import make_split_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed train/val/test split.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--class_mapping", required=True)
    parser.add_argument("--strategy", default="iterative_stratification")
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.strategy != "iterative_stratification":
        raise ValueError("Only iterative_stratification is supported in this implementation")
    df = make_split_from_files(args.labels, args.class_mapping, args.out, seed=args.seed)
    print(df["split"].value_counts().to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
