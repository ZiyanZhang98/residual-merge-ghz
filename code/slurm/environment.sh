#!/bin/bash
# Sourced by batch scripts; defines qn_command.
# Shared environment setup for the Slurm scripts (written for Snellius; adapt module names to your cluster).
# Match the existing Snellius scripts; set QN_MODULES="" to skip module loading.
qn_module_names="${QN_MODULES-2023 CUDA/12.4.0}"
if [[ -n "$qn_module_names" ]]; then
    read -r -a qn_modules <<< "$qn_module_names"
    module load "${qn_modules[@]}"
fi
qn_env="${QN_ENV:-llm}"
if [[ -n "${QN_PYTHON:-}" ]]; then
    command -v "$QN_PYTHON" >/dev/null || { echo "QN_PYTHON is not executable: $QN_PYTHON" >&2; exit 2; }
    qn_command=("$QN_PYTHON")
elif [[ "${CONDA_DEFAULT_ENV:-}" == "$qn_env" ]]; then
    command -v python >/dev/null
    qn_command=(python)
elif command -v micromamba >/dev/null; then
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    qn_command=(micromamba run -n "$qn_env" python)
elif command -v mamba >/dev/null; then
    qn_command=(mamba run -n "$qn_env" python)
elif command -v conda >/dev/null; then
    qn_command=(conda run --no-capture-output -n "$qn_env" python)
else
    echo "Activate the existing $qn_env environment, expose micromamba/mamba/conda, or set QN_PYTHON to its Python executable." >&2
    exit 2
fi

# Preserve the original single-thread numerical protocol; 18 cores is the A100 allocation unit.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Snellius user storage policy: never fall back to system /tmp or /var/tmp.
: "${TMPDIR:?Slurm must provide user scratch in TMPDIR}"
case "$TMPDIR" in
    /tmp|/tmp/*|/var/tmp|/var/tmp/*)
        echo "Use Snellius scratch via TMPDIR, not system temporary directories" >&2
        exit 2 ;;
esac
qn_cache="$TMPDIR/qn_h3_sweep_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$qn_cache/matplotlib" CUDA_CACHE_PATH="$qn_cache/cuda" XDG_CACHE_HOME="$qn_cache/xdg"
mkdir -p "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"

