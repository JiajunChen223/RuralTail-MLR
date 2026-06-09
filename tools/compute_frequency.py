from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from ruraltail_mlr.data.frequency import compute_frequency_from_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute class frequency and Head/Medium/Tail groups.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--class_mapping", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    freq, hmt = compute_frequency_from_files(args.labels, args.split, args.class_mapping, args.out_dir)
    print(freq.to_string(index=False))
    print(hmt)


if __name__ == "__main__":
    main()
