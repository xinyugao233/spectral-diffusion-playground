#!/bin/bash

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"

if [ "$#" -ne 2 ]; then
  echo "usage: e009_training_entrypoint.sh CONFIG {fresh|resume}" >&2
  exit 1
fi
case "$E009_REPO_ROOT" in
  /*) ;;
  *) echo "error: E009_REPO_ROOT must be absolute" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
CONFIG="$1"
MODE="$2"
case "$CONFIG" in
  /*) ;;
  *) CONFIG="${REPO_ROOT}/${CONFIG}" ;;
esac
if [ "$MODE" != fresh ] && [ "$MODE" != resume ]; then
  echo "error: mode must be fresh or resume" >&2
  exit 1
fi

PREFLIGHT="${REPO_ROOT}/scripts/e009_preflight.py"
WRAPPER="/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py"
for path in "$PREFLIGHT" "$CONFIG" "$WRAPPER" "${REPO_ROOT}/.git"; do
  if [ ! -e "$path" ]; then
    echo "error: required path is missing: ${path}" >&2
    exit 1
  fi
done

cd "$REPO_ROOT"
python "$PREFLIGHT" \
  --repo-root "$REPO_ROOT" \
  --expected-repo-commit "$E009_REPO_COMMIT" \
  --config "$CONFIG" \
  --mode "$MODE"

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE=provided
else
  ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
  ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-single}"
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
  TMP_SOURCE=fallback
  mkdir -p "$TMP_ROOT"
fi
if [ ! -d "$TMP_ROOT" ] || [ ! -w "$TMP_ROOT" ]; then
  echo "error: temporary directory is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
export SLURM_TMPDIR="$(cd "$TMP_ROOT" && pwd -P)"

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
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E009_REPO_COMMIT}"
echo "config=${CONFIG}"
echo "mode=${MODE}"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "slurm_tmpdir_writable=true"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="/home/xggh8/projects/zw-lab/src:/home/xggh8/projects/zw-lab:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

python "$WRAPPER" --config "$CONFIG"
