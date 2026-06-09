from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from ruraltail_mlr.data.audit import run_data_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit China-MAS-50k data and build processed files.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--no_check_images", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--label_direction_verified", action="store_true")
    parser.add_argument("--label_direction_note", default="")
    args = parser.parse_args()
    result = run_data_audit(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        check_images=not args.no_check_images,
        checksum=args.checksum,
        label_direction_verified=args.label_direction_verified,
        label_direction_note=args.label_direction_note,
    )
    print(f"labels_clean: {result.labels_clean}")
    print(f"images_index: {result.images_index}")
    print(f"class_mapping: {result.class_mapping}")
    print(f"audit_md: {result.audit_md}")


if __name__ == "__main__":
    main()
