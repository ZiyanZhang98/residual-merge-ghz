#!/bin/bash
# HANDOFF 22.28 M3: deployment of L and C1 on the resolved n=4 development tasks, sharded by task.
# Submit from the repository root (at most 4 concurrent A100s):
#   sbatch --array=0-<S-1>%4 --time=HH:MM:SS --export=ALL,QN_RM_TRAIN_JOB=<train job>,QN_RM_SHARDS=<S>,QN_RM_MINUTES=<M> \
#          code/slurm/deploy.sbatch
#SBATCH --job-name=qn_rm_dep
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_dep-%A_%a.out
#SBATCH --error=logs/qn_rm_dep-%A_%a.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM_TRAIN_JOB:?train job id}"
: "${QN_RM_SHARDS:?number of shards}"
: "${QN_RM_MINUTES:?python deadline in minutes, below --time}"
qn_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run.py" deploy --device cuda \
  --minutes "$QN_RM_MINUTES" --train-job "$QN_RM_TRAIN_JOB" --out "$qn_rm/results/deploy_${qn_tag}" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$QN_RM_SHARDS"
