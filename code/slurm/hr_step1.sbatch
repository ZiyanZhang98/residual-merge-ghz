#!/bin/bash
# GPU stages of HANDOFF 22.17 (smoke | step1 | step2). Submit from the repository root (see README.md).
# One A100 = the Snellius quarter-node unit (18 cores, proportional memory); arrays are throttled
# to at most 4 concurrent GPUs because the account shares a 16-node QOS limit.
#SBATCH --job-name=qn_hr
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=qn_hr-%x-%A_%a.out
#SBATCH --error=qn_hr-%x-%A_%a.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_hr="$qn_repo/code/frozen/hard_regime"
source "$qn_repo/code/slurm/environment.sh"
: "${QN_HR_MODE:?smoke|step1|step2}" "${QN_HR_MINUTES:?python deadline in minutes, below --time}"
qn_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
qn_args=("$QN_HR_MODE" --device cuda --minutes "$QN_HR_MINUTES" --out "$qn_hr/results/${QN_HR_MODE}_${qn_tag}"
  --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "${QN_HR_SHARDS:-1}")
[[ -n "${QN_HR_STEP1:-}" ]] && qn_args+=(--step1-job "$QN_HR_STEP1")
[[ -n "${QN_HR_DECIDE1:-}" ]] && qn_args+=(--decide1-job "$QN_HR_DECIDE1")
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_hr/run.py" "${qn_args[@]}"
