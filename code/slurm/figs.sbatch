#!/bin/bash
# HANDOFF 22.40: paper figures and summaries from existing cells (CPU, read-only; no simulation, training or truth).
# Submit from the repository root:
#   sbatch code/slurm/figs.sbatch
#SBATCH --job-name=qn_rm_figs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_figs-%j.out
#SBATCH --error=logs/qn_rm_figs-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=4 "${qn_command[@]}" "$qn_rm/run2.py" figs --device cpu --minutes 27 \
  --out "$qn_rm/results/figs_$SLURM_JOB_ID"
