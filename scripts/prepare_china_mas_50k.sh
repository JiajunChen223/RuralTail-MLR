#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-data/raw/China-MAS-50k.7z}"

cd "$(dirname "$0")/.."

if ! command -v 7z >/dev/null 2>&1; then
  echo "7z is required. Install p7zip-full first, e.g. apt-get update && apt-get install -y p7zip-full" >&2
  exit 1
fi

mkdir -p data/raw/china_mas_50k data/processed
7z x -y "$ARCHIVE" -odata/raw/china_mas_50k

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No Python executable found. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" tools/prepare_china_mas_50k.py \
  --raw_root data/raw/china_mas_50k \
  --processed_dir data/processed
