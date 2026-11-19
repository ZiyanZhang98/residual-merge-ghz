#!/bin/bash
# HANDOFF 22.38: the registered n=3 analysis (CPU, read-only); runs once.
# Submit from the repository root:
#   sbatch --dependency=afterok:<eval> --export=ALL,QN_RM2_JOB=<eval>,QN_RM2_TRAIN=<train> code/slurm/n3_cpu.sbatch
#SBATCH --job-name=qn_rm_n3a
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_n3a-%j.out
#SBATCH --error=logs/qn_rm_n3a-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM2_JOB:?n=3 evaluation array id}"
: "${QN_RM2_TRAIN:?training job id}"
srun --ntasks=1 --cpus-per-task=4 "${qn_command[@]}" "$qn_rm/run2.py" n3_analyze --device cpu --minutes 27 \
  --job "$QN_RM2_JOB" --train-job "$QN_RM2_TRAIN" --out "$qn_rm/results/n3a_$SLURM_JOB_ID"
