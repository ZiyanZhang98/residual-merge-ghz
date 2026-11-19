#!/bin/bash
# HANDOFF 22.37: Monte Carlo truth and the frozen L, C1, F2 grids on the test tasks, sharded by task (<= 4 A100s).
# Submit from the repository root after the pool job:
#   sbatch --array=0-11%4 --dependency=afterok:<pool> --export=ALL,QN_RM2_SHARDS=12,QN_RM2_POOL=<pool> code/slurm/test.sbatch
#SBATCH --job-name=qn_rm_test
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_test-%A_%a.out
#SBATCH --error=logs/qn_rm_test-%A_%a.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM2_SHARDS:?number of shards}"
: "${QN_RM2_POOL:?test pool job id}"
qn_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" test_run --device cuda --minutes 235 \
  --pool-job "$QN_RM2_POOL" --out "$qn_rm/results/test_${qn_tag}" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$QN_RM2_SHARDS"
