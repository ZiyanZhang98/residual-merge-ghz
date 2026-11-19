#!/bin/bash
# HANDOFF 22.38: train the n=3 pairwise residual cost model (frozen m2 settings, 1 seed); label-level evaluation and
# label noise report. Submit from the repository root after the label array:
#   sbatch --dependency=afterok:<labels> --export=ALL,QN_RM2_JOB=<labels> code/slurm/n3_train.sbatch
#SBATCH --job-name=qn_rm_n3train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_n3train-%j.out
#SBATCH --error=logs/qn_rm_n3train-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM2_JOB:?label array id}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" n3_train --device cuda --minutes 57 \
  --job "$QN_RM2_JOB" --out "$qn_rm/results/n3train_$SLURM_JOB_ID"
