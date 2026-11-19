#!/bin/bash
# Paper figures for the comparison experiments and the cause figure (n=4). CPU, read-only.
# Submit from the repository root:
#   sbatch code/slurm/cmpfig.sbatch
#SBATCH --job-name=qn_rm_cmpfig
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=logs/qn_rm_cmpfig-%j.out
#SBATCH --error=logs/qn_rm_cmpfig-%j.err
set -Eeuo pipefail
qn_repo="${SLURM_SUBMIT_DIR:?}"
qn_rm="$qn_repo/code/residual_merge"
[[ -f "$qn_rm/rm_cmpfig.py" ]] || { echo "Submit from the repository root" >&2; exit 2; }
QN_MODULES="${QN_MODULES-2023}"
source "$qn_repo/code/slurm/environment.sh"
out="$qn_rm/results/cmpfig_$SLURM_JOB_ID"
srun --ntasks=1 --cpus-per-task=2 "${qn_command[@]}" "$qn_rm/rm_cmpfig.py" --out "$out"
mkdir -p "$qn_repo/paper_figures"
srun --ntasks=1 --cpus-per-task=2 "${qn_command[@]}" "$qn_rm/rm_causefig.py" --out "$qn_rm/results/causefig_$SLURM_JOB_ID"
cp "$out"/figA_cost_ladder{,_a,_b}.{pdf,png} "$out"/figB_L_vs_baselines{,_a,_b}.{pdf,png} "$out"/figE_alpha_sweep.{pdf,png} "$out"/figD_error_vs_atoms.{pdf,png} "$out"/figR_ratio.{pdf,png} "$qn_rm/results/causefig_$SLURM_JOB_ID"/fig_cause_pta_repair_{a,b}.{pdf,png} "$qn_repo/paper_figures/"
# Recompile the pipeline-figure preview (TikZ source in paper/figures; the paper itself \input's fig_pipeline.tex).
( cd "$qn_repo/paper_figures" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMPDIR" fig_pipeline_preview.tex >/dev/null && cp "$TMPDIR"/fig_pipeline_preview.pdf . )
