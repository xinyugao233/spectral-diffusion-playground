#!/usr/bin/env bash
#SBATCH --job-name=e006-check
#SBATCH --partition=general
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}"
mkdir -p "$SLURM_TMPDIR"
export TMPDIR="$SLURM_TMPDIR/tmp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_HOME="$SLURM_TMPDIR/torch_cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export PIP_CACHE_DIR="$SLURM_TMPDIR/pip-cache"

PROJECT_ROOT="${E006_PROJECT_ROOT:-${HOME}/projects/spectral-diffusion-playground}"
test -d "$PROJECT_ROOT/.git"
export PYTHONPATH="${PROJECT_ROOT}/src"

find "$PROJECT_ROOT/src" "$PROJECT_ROOT/experiments" "$PROJECT_ROOT/tests" \
  -name '*.py' -print0 | xargs -0 python -m py_compile

python -c \
  "import spectral_diffusion_playground.denoiser_trajectory as module; assert len(module.TIMESTEPS) == 41"

printf 'Experiment 6 Slurm smoke check passed at commit %s\n' \
  "$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
