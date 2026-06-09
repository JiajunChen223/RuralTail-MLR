#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
GPU_LIST="${GPU_LIST:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-}"
ALLOW_LEGACY="${ALLOW_LEGACY:-}"

mkdir -p artifacts

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_SHARDS="${#GPUS[@]}"
declare -a PIDS=()

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  args=(
    tools/run_supplement_group_eval.py
    --shard-index "$idx"
    --num-shards "$NUM_SHARDS"
    --num-workers "$NUM_WORKERS"
    "$@"
  )
  if [[ -n "$BATCH_SIZE" ]]; then
    args+=(--batch-size "$BATCH_SIZE")
  fi
  if [[ -n "$ALLOW_LEGACY" ]]; then
    args+=("$ALLOW_LEGACY")
  fi
  echo "[supplement] GPU ${gpu}: shard ${idx}/${NUM_SHARDS}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "${args[@]}" \
    > "artifacts/supplement_group_eval_gpu${gpu}.log" 2>&1 &
  PIDS[$idx]=$!
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  echo "At least one evaluation shard failed. Inspect artifacts/supplement_group_eval_gpu*.log." >&2
  exit "$status"
fi

"$PYTHON" tools/run_supplement_group_eval.py "$@" --summarize-only
echo "Supplement outputs written by tools/run_supplement_group_eval.py"
