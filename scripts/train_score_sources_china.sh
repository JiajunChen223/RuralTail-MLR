#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"

GPU_LIST="${GPU_LIST:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-2026}"
RUN_PREFIX="${RUN_PREFIX:-paper_china_seed${SEED}}"
LOG_DIR="${LOG_DIR:-artifacts/run_logs/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
JOB_INDICES="${JOB_INDICES:-}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty. Example: GPU_LIST=0,1,2 bash scripts/train_score_sources_china.sh" >&2
  exit 2
fi

"${PYTHON}" tools/check_environment.py --require_cuda --require_timm --require_mamba_ssm --require_causal_conv1d

COMMON_ARGS=("run.seed=${SEED}" "train.num_workers=${NUM_WORKERS}")

JOBS=(
  "resnet50_bce|experiment=main_resnet50 loss=bce"
  "resnet50_focal|experiment=main_resnet50 loss=focal"
  "resnet50_asl|experiment=main_resnet50 loss=asl"
  "resnet50_talc|experiment=main_resnet50 +method=asl_softf1_t"
  "efficientnetv2_s_bce|experiment=main_efficientnetv2_s loss=bce"
  "efficientnetv2_s_focal|experiment=main_efficientnetv2_s loss=focal"
  "efficientnetv2_s_asl|experiment=main_efficientnetv2_s loss=asl"
  "efficientnetv2_s_talc|experiment=main_efficientnetv2_s +method=asl_softf1_t"
  "pvtv2_b2_bce|experiment=main_pvtv2_b2 loss=bce"
  "pvtv2_b2_focal|experiment=main_pvtv2_b2 loss=focal"
  "pvtv2_b2_asl|experiment=main_pvtv2_b2 loss=asl"
  "pvtv2_b2_talc|experiment=main_pvtv2_b2 +method=asl_softf1_t"
  "mambaout_s_bce|experiment=main_mambaout_s loss=bce"
  "mambaout_s_focal|experiment=main_mambaout_s loss=focal"
  "mambaout_s_asl|experiment=main_mambaout_s loss=asl"
  "mambaout_s_talc|experiment=main_mambaout_s +method=asl_softf1_t"
  "recent_sfin_resnet18_bce|experiment=main_sfin_resnet18 loss=bce"
  "recent_sfin_resnet18_focal|experiment=main_sfin_resnet18 loss=focal"
  "recent_sfin_resnet18_asl|experiment=main_sfin_resnet18 loss=asl"
  "recent_sfin_resnet18_talc|experiment=main_sfin_resnet18 +method=asl_softf1_t"
  "recent_mlmamba_resnet18_bce|experiment=main_mlmamba_resnet18 loss=bce"
  "recent_mlmamba_resnet18_focal|experiment=main_mlmamba_resnet18 loss=focal"
  "recent_mlmamba_resnet18_asl|experiment=main_mlmamba_resnet18 loss=asl"
  "recent_mlmamba_resnet18_talc|experiment=main_mlmamba_resnet18 +method=asl_softf1_t"
)

SELECTED_JOBS=()
if [[ -n "${JOB_INDICES}" ]]; then
  IFS=',' read -r -a IDX_LIST <<< "${JOB_INDICES}"
  for idx in "${IDX_LIST[@]}"; do
    [[ -z "${idx}" ]] && continue
    SELECTED_JOBS+=("${JOBS[$idx]}")
  done
else
  SELECTED_JOBS=("${JOBS[@]}")
fi

run_task() {
  local gpu="$1"
  local job_name="$2"
  shift 2
  local run_name="${RUN_PREFIX}_${job_name}"
  local log_path="${LOG_DIR}/${job_name}.log"
  echo "[gpu${gpu}] START ${job_name} $(date -Is)" | tee -a "${LOG_DIR}/queue.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" tools/train.py "$@" "run.name=${run_name}" "${COMMON_ARGS[@]}" 2>&1 | tee "${log_path}"
  echo "[gpu${gpu}] DONE  ${job_name} $(date -Is)" | tee -a "${LOG_DIR}/queue.log"
}

status=0
declare -a pids=()
for idx in "${!GPUS[@]}"; do
  (
    gpu="${GPUS[$idx]}"
    job_idx="$idx"
    while (( job_idx < ${#SELECTED_JOBS[@]} )); do
      spec="${SELECTED_JOBS[$job_idx]}"
      job_name="${spec%%|*}"
      args_string="${spec#*|}"
      read -r -a job_args <<< "$args_string"
      run_task "$gpu" "$job_name" "${job_args[@]}"
      ((job_idx += ${#GPUS[@]}))
    done
  ) &
  pids[$idx]=$!
done
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
