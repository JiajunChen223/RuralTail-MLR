#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-2026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/supplement_group_eval_seed${SEED}}"
TABLE_DIR="${TABLE_DIR:-artifacts/paper_tables}"

if [[ ! -f "${OUTPUT_ROOT}/test_results_by_carrier.csv" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT" SEED="$SEED" PYTHON="$PYTHON" bash scripts/evaluate_paper_protocol.sh
fi

"${PYTHON}" tools/make_paper_tables.py --eval-root "$OUTPUT_ROOT" --out-dir "$TABLE_DIR"
