#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TAR_PATH="${1:-data/raw/Agriculture-Vision-2021.tar.gz}"
RAW_ROOT="${2:-data/raw/agriculture_vision_2021/Agriculture-Vision-2021}"
OUT_DIR="${3:-data/processed/agriculture_vision_2021}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-32}"
RAW_PARENT="$(dirname "${RAW_ROOT}")"

mkdir -p "${RAW_PARENT}"

echo "[1/2] Extracting labeled train/val RGB images and label masks into ${RAW_PARENT}"
tar -xzf "${TAR_PATH}" -C "${RAW_PARENT}" \
  --wildcards \
  'Agriculture-Vision-2021/train/images/rgb/*' \
  'Agriculture-Vision-2021/val/images/rgb/*' \
  'Agriculture-Vision-2021/train/labels/*' \
  'Agriculture-Vision-2021/val/labels/*'

echo "[2/2] Building RuralTail-MLR metadata with ${JOBS} workers"
"${PYTHON_BIN}" tools/prepare_agriculture_vision.py \
  --raw_root "${RAW_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --seed 20260501 \
  --ratios 0.8,0.1,0.1 \
  --min_positive_pixels 1 \
  --jobs "${JOBS}"
