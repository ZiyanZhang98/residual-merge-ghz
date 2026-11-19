#!/bin/bash
# Paper figures: PTA simulator vs. Monte Carlo truth (P and F per round), n=4 test set. CPU, read-only.
# Submit from the repository root:
#   sbatch code/slurm/ptafig.sbatch [FAMILY]   (paper-version family, default ABB)
#SBATCH --job-name=qn_rm_ptafig
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_ptafig-%j.out
#SBATCH --error=logs/qn_rm_ptafig-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/rm_ptafig.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
fam="${1:-ABB}"
out="$qn_rm/results/ptafig_$SLURM_JOB_ID"
srun --ntasks=1 --cpus-per-task=2 "${qn_command[@]}" "$qn_rm/rm_ptafig.py" --main-family "$fam" --out "$out"
mkdir -p "$qn_repo/paper_figures"
cp "$out"/fig_pta_truth_${fam}_{P,F}.{pdf,png} "$qn_repo/paper_figures/"
