#!/bin/bash
# HANDOFF 22.28 M0 smoke (residual merge). Submit from the repository root:
#   sbatch code/slurm/run.sbatch
# One A100 = the Snellius quarter-node unit (18 cores, proportional memory). Python runs only here, on the
# compute node; caches go to the job's TMPDIR (environment.sh refuses /tmp); results stay in project space.
#SBATCH --job-name=qn_rm
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm-%j.out
#SBATCH --error=logs/qn_rm-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM_MODE:=smoke}"
: "${QN_RM_MINUTES:=40}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run.py" "$QN_RM_MODE" --device cuda \
  --minutes "$QN_RM_MINUTES" --out "$qn_rm/results/${QN_RM_MODE}_${SLURM_JOB_ID}"
