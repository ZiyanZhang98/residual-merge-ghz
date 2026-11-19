#!/bin/bash
# HANDOFF 22.37: generate the test preparations and tasks (specifications only, CPU) and check their isolation.
# Submit from the repository root:
#   sbatch code/slurm/test_pool.sbatch
#SBATCH --job-name=qn_rm_tpool
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_tpool-%j.out
#SBATCH --error=logs/qn_rm_tpool-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
srun --ntasks=1 --cpus-per-task=2 "${qn_command[@]}" "$qn_rm/run2.py" test_pool --device cpu --minutes 12 \
  --out "$qn_rm/results/testpool_$SLURM_JOB_ID"
