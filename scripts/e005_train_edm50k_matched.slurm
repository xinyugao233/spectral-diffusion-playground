#!/bin/bash
#SBATCH --job-name=e005_edm50k
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e005-edm50k-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e005-edm50k-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

set -euo pipefail

if [ "${SLURM_JOB_ID:-}" = "" ]; then
  echo "Refusing to train outside Slurm" >&2
  exit 1
fi

if [ "$#" -ne 2 ]; then
  echo "Usage: sbatch scripts/e005_train_edm50k_matched.slurm CONFIG {fresh|resume}" >&2
  exit 1
fi

CONFIG="$1"
MODE="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/configs/e005_edm50k_matched_40000kimg_manifest.json"
WRAPPER="/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py"

if [ "$MODE" != "fresh" ] && [ "$MODE" != "resume" ]; then
  echo "Mode must be fresh or resume" >&2
  exit 1
fi

python "${REPO_ROOT}/scripts/e005_preflight.py" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --mode "$MODE"

test -n "${SLURM_TMPDIR:-}"
export TMPDIR="${SLURM_TMPDIR}/tmp"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/.cache"
export TORCH_HOME="${SLURM_TMPDIR}/torch_cache"
export MPLCONFIGDIR="${SLURM_TMPDIR}/matplotlib"
export WANDB_DIR="${SLURM_TMPDIR}/wandb"
export HF_HOME="${SLURM_TMPDIR}/hf"
export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf/transformers"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf/datasets"
export PIP_CACHE_DIR="${SLURM_TMPDIR}/pip-cache"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX"

export PYTHONPATH="/home/xggh8/projects/zw-lab/src:/home/xggh8/projects/zw-lab:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

echo "training_mode=${MODE}"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"
echo "expected_output=/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg"

python "$WRAPPER" --config "$CONFIG"
