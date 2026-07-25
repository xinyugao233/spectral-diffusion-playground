#!/usr/bin/env bash
#SBATCH --job-name=e006-fixed-model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

export TMPDIR="$SLURM_TMPDIR/tmp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_HOME="$SLURM_TMPDIR/torch_cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export HF_HOME="$SLURM_TMPDIR/hf"
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf/transformers"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf/datasets"
export PIP_CACHE_DIR="$SLURM_TMPDIR/pip-cache"

mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

PROJECT_ROOT="${HOME}/projects/spectral-diffusion-playground"
UPSTREAM_ROOT="${HOME}/projects/guided-diffusion"
CHECKPOINT_PATH="${HOME}/data/spectral-diffusion-playground/models/256x256_diffusion_uncond.pt"
RUN_ID="experiment_06_${SLURM_JOB_ID}"
TEMP_ROOT="${SLURM_TMPDIR}/${RUN_ID}"
FINAL_ROOT="${HOME}/data/spectral-diffusion-playground/experiment_06/${RUN_ID}"

test -d "$PROJECT_ROOT/.git"
test -d "$UPSTREAM_ROOT/.git"
test -f "$CHECKPOINT_PATH"
test "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" = \
  "22e0df8183507e13a7813f8d38d51b072ca1e67c"

mkdir -p "$TEMP_ROOT/results" "$TEMP_ROOT/figures" "$TEMP_ROOT/logs"
export PYTHONPATH="${PROJECT_ROOT}/src"

python "$PROJECT_ROOT/scripts/validate_natural_image_dataset.py"
python -m unittest discover -s "$PROJECT_ROOT/tests"

python "$PROJECT_ROOT/experiments/06_denoiser_trajectory.py" \
  --guided-diffusion-root "$UPSTREAM_ROOT" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --results-dir "$TEMP_ROOT/results" \
  --output-dir "$TEMP_ROOT/figures" \
  --batch-size 2 \
  2>&1 | tee "$TEMP_ROOT/logs/experiment_06.log"

mkdir -p "$FINAL_ROOT"
cp -R "$TEMP_ROOT/results" "$FINAL_ROOT/results"
cp -R "$TEMP_ROOT/figures" "$FINAL_ROOT/figures"
cp -R "$TEMP_ROOT/logs" "$FINAL_ROOT/logs"

printf 'Experiment 6 staged at %s\n' "$FINAL_ROOT"
