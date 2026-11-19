#!/bin/bash
# HANDOFF 22.28 M2: train the pairwise residual cost model (1 seed), label-level evaluation, deployment profile.
# Submit from the repository root:  sbatch code/slurm/train.sbatch
#SBATCH --job-name=qn_rm_train
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_train-%j.out
#SBATCH --error=logs/qn_rm_train-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run.py" train --device cuda --minutes 85 \
  --out "$qn_rm/results/train_$SLURM_JOB_ID"
