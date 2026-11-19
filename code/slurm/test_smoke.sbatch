#!/bin/bash
# HANDOFF 22.37: smoke of the independent test set (development tasks only; no test truth or method runs).
# Submit from the repository root:
#   sbatch code/slurm/test_smoke.sbatch
#SBATCH --job-name=qn_rm_tsmk
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_tsmk-%j.out
#SBATCH --error=logs/qn_rm_tsmk-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run2.py" test_smoke --device cuda --minutes 42 \
  --out "$qn_rm/results/testsmoke_$SLURM_JOB_ID"
