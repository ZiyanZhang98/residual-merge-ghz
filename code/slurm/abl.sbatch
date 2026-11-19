#!/bin/bash
# HANDOFF 22.39: label-level ablations of the frozen L (4 trainings + 2 evaluations, n=4 labels). Submit from the
# repository root:
#   sbatch code/slurm/abl.sbatch
#SBATCH --job-name=qn_rm_abl
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_abl-%j.out
#SBATCH --error=logs/qn_rm_abl-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" abl --device cuda --minutes 87 \
  --out "$qn_rm/results/abl_$SLURM_JOB_ID"
