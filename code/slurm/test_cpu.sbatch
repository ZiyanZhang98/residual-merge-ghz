#!/bin/bash
# HANDOFF 22.37: the registered test-set analysis (CPU, read-only on the pool and test shards); runs once.
# Submit from the repository root:
#   sbatch --dependency=afterok:<array> --export=ALL,QN_RM2_JOB=<array>,QN_RM2_POOL=<pool> code/slurm/test_cpu.sbatch
#SBATCH --job-name=qn_rm_testa
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_testa-%j.out
#SBATCH --error=logs/qn_rm_testa-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/run2.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
: "${QN_RM2_JOB:?test deployment array id}"
: "${QN_RM2_POOL:?test pool job id}"
srun --ntasks=1 --cpus-per-task=4 "${qn_command[@]}" "$qn_rm/run2.py" test_analyze --device cpu --minutes 27 \
  --job "$QN_RM2_JOB" --pool-job "$QN_RM2_POOL" --out "$qn_rm/results/testa_$SLURM_JOB_ID"
