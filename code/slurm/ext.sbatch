#!/bin/bash
# HANDOFF 22.35: extended evaluation grids (frozen L, C1, F2) on all resolved n=4 M3 tasks, sharded by task (<= 4 A100s).
# Submit from the repository root:
#   sbatch --array=0-3%4 --export=ALL,QN_RM2_SHARDS=4 code/slurm/ext.sbatch
#SBATCH --job-name=qn_rm_ext
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_ext-%A_%a.out
#SBATCH --error=logs/qn_rm_ext-%A_%a.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM2_SHARDS:?number of shards}"
qn_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" ext_deploy --device cuda --minutes 85 \
  --out "$qn_rm/results/ext_${qn_tag}" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$QN_RM2_SHARDS"
