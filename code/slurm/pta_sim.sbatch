#!/bin/bash
# HANDOFF 22.41: PTA simulator on every resolved development, test and n=3 task (CPU, read-only; no new truth or training).
# Submit from the repository root:
#   sbatch code/slurm/pta_sim.sbatch
#SBATCH --job-name=qn_rm_pta
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_pta-%j.out
#SBATCH --error=logs/qn_rm_pta-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=4 "${qn_command[@]}" "$qn_rm/run2.py" pta_sim --device cpu --minutes 57 \
  --out "$qn_rm/results/pta_$SLURM_JOB_ID"
