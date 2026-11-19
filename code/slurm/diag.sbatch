#!/bin/bash
# HANDOFF 22.28 development diagnostics of a trained model (per-compression merge log, per-prefix errors).
# Submit from the repository root:  sbatch --export=ALL,QN_RM_TRAIN_JOB=<train job> code/slurm/diag.sbatch
#SBATCH --job-name=qn_rm_diag
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_diag-%j.out
#SBATCH --error=logs/qn_rm_diag-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM_TRAIN_JOB:?train job id}"
srun --ntasks=1 --cpus-per-task=18 "${qn_command[@]}" "$qn_rm/run.py" diagnose --device cuda --minutes 40 \
  --train-job "$QN_RM_TRAIN_JOB" --out "$qn_rm/results/diagnose_$SLURM_JOB_ID"
