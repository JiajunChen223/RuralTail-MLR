#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-2026}"
RUNS_DIR="${RUNS_DIR:-outputs/runs}"
MANIFEST="${MANIFEST:-artifacts/eval_manifest_seed${SEED}.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/supplement_group_eval_seed${SEED}}"
GPU_LIST="${GPU_LIST:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"${PYTHON}" tools/build_eval_manifest.py --runs "$RUNS_DIR" --out "$MANIFEST" --seed "$SEED"
GPU_LIST="$GPU_LIST" NUM_WORKERS="$NUM_WORKERS" PYTHON="$PYTHON" \
  bash scripts/run_supplement_group_eval.sh --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT"
