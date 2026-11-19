#!/bin/bash
# HANDOFF 22.38: smoke of the n=3 second setting (n=4 label reproduction, n=3 labels/training/runner on one task).
# Submit from the repository root:  sbatch code/slurm/n3_smoke.sbatch
#SBATCH --job-name=qn_rm_n3smk
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_n3smk-%j.out
#SBATCH --error=logs/qn_rm_n3smk-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" n3_smoke --device cuda --minutes 57 \
  --out "$qn_rm/results/n3smoke_$SLURM_JOB_ID"
