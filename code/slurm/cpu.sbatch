#!/bin/bash
# HANDOFF 22.28 CPU modes, read-only on earlier artifacts. Submit from the repository root:
#   M1 label summary: sbatch --export=ALL,QN_RM_CPU_MODE=labels_report,QN_RM_LABELS_JOB=<label array> code/slurm/cpu.sbatch
#   M3 analysis:      sbatch --export=ALL,QN_RM_CPU_MODE=analyze,QN_RM_DEPLOY_JOB=<deploy array> code/slurm/cpu.sbatch
#SBATCH --job-name=qn_rm_cpu
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_cpu-%j.out
#SBATCH --error=logs/qn_rm_cpu-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM_CPU_MODE:=labels_report}"
extra=()
if [[ "$QN_RM_CPU_MODE" == labels_report ]]; then
  : "${QN_RM_LABELS_JOB:?label array job id}"; extra=(--labels-job "$QN_RM_LABELS_JOB")
else
  : "${QN_RM_DEPLOY_JOB:?deploy array job id}"; extra=(--deploy-job "$QN_RM_DEPLOY_JOB")
fi
srun --ntasks=1 --cpus-per-task=4 "${qn_command[@]}" "$qn_rm/run.py" "$QN_RM_CPU_MODE" --device cpu --minutes 27 \
  "${extra[@]}" --out "$qn_rm/results/${QN_RM_CPU_MODE}_$SLURM_JOB_ID"
