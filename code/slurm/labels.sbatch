#!/bin/bash
# HANDOFF 22.28 M1: pairwise residual-merge labels, n=4 fit + monitor preparations, sharded by preparation.
# Submit from the repository root (at most 4 concurrent A100s; the account shares a 16-node QOS limit):
#   sbatch --array=0-11%4 --export=ALL,QN_RM_SHARDS=12 code/slurm/labels.sbatch
#SBATCH --job-name=qn_rm_lab
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_lab-%A_%a.out
#SBATCH --error=logs/qn_rm_lab-%A_%a.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM_SHARDS:?number of shards}"
: "${QN_RM_MINUTES:=85}"
qn_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run.py" "${QN_RM_LABEL_MODE:-labels}" --device cuda \
  --minutes "$QN_RM_MINUTES" --out "$qn_rm/results/${QN_RM_LABEL_MODE:-labels}_${qn_tag}" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$QN_RM_SHARDS"
