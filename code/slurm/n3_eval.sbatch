#!/bin/bash
# HANDOFF 22.38: L, C1, F2 (+M0, F2mul) on the frozen extended grids, resolved n=3 monitor tasks, sharded by task.
# Submit from the repository root after the training job (<= 4 A100s):
#   sbatch --array=0-7%4 --dependency=afterok:<train> --export=ALL,QN_RM2_SHARDS=8,QN_RM2_TRAIN=<train> code/slurm/n3_eval.sbatch
#SBATCH --job-name=qn_rm_n3eval
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_n3eval-%A_%a.out
#SBATCH --error=logs/qn_rm_n3eval-%A_%a.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM2_SHARDS:?number of shards}"
: "${QN_RM2_TRAIN:?training job id}"
qn_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" n3_eval --device cuda --minutes 175 \
  --train-job "$QN_RM2_TRAIN" --out "$qn_rm/results/n3eval_${qn_tag}" --index "${SLURM_ARRAY_TASK_ID:-0}" --shards "$QN_RM2_SHARDS"
