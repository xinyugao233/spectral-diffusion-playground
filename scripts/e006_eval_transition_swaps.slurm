#!/bin/bash
#SBATCH --job-name=e006_swaps
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e006-swaps-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e006-swaps-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "error: SLURM_JOB_ID is unset" >&2
  exit 1
fi
if [ -z "${E006_REPO_ROOT:-}" ]; then
  echo "error: E006_REPO_ROOT is required" >&2
  exit 1
fi
if [ -z "${E006_REPO_COMMIT:-}" ]; then
  echo "error: E006_REPO_COMMIT is required" >&2
  exit 1
fi
case "$E006_REPO_ROOT" in
  /*) ;;
  *)
    echo "error: E006_REPO_ROOT must be absolute: ${E006_REPO_ROOT}" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$E006_REPO_ROOT" && pwd -P)"
ENTRYPOINT="${REPO_ROOT}/experiments/06_transition_window_swaps.py"
CONFIG="${REPO_ROOT}/configs/e006_transition_window_swaps.json"
EXPECTED_CONFIG_SHA256="333a4853733d08e2e289a3bbef6f01ef083834e12e31bfd08b797e0fb37ffec2"

if [ ! -d "${REPO_ROOT}/.git" ]; then
  echo "error: expected Git repository at ${REPO_ROOT}" >&2
  exit 1
fi
if [ ! -f "$ENTRYPOINT" ] || [ ! -f "$CONFIG" ]; then
  echo "error: E006 entrypoint or config missing under ${REPO_ROOT}" >&2
  exit 1
fi
ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$E006_REPO_COMMIT" ]; then
  echo "error: repository commit mismatch: ${ACTUAL_COMMIT} != ${E006_REPO_COMMIT}" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]; then
  echo "error: repository worktree is not clean" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 1
fi
ACTUAL_CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
if [ "$ACTUAL_CONFIG_SHA256" != "$EXPECTED_CONFIG_SHA256" ]; then
  echo "error: config SHA-256 mismatch" >&2
  echo "observed=${ACTUAL_CONFIG_SHA256}" >&2
  echo "expected=${EXPECTED_CONFIG_SHA256}" >&2
  exit 1
fi

MODE="${1:-full}"
case "$MODE" in
  full)
    OUTPUT_DIR="/home/xggh8/data/zw-lab/e006_transition_window_swaps"
    ;;
  smoke)
    OUTPUT_DIR="/home/xggh8/scratch/zw-lab/e006_transition_window_swaps_smoke_${SLURM_JOB_ID}"
    ;;
  preflight)
    OUTPUT_DIR=""
    ;;
  *)
    echo "error: usage: sbatch $0 {preflight|smoke|full}" >&2
    exit 1
    ;;
esac

if [ -n "$OUTPUT_DIR" ] && [ -e "$OUTPUT_DIR" ]; then
  echo "error: output path already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE="provided"
else
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e006_${SLURM_JOB_ID}"
  TMP_SOURCE="fallback"
fi
mkdir -p "$TMP_ROOT"
if [ ! -d "$TMP_ROOT" ] || [ ! -w "$TMP_ROOT" ]; then
  echo "error: E006 temporary directory is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
export SLURM_TMPDIR="$TMP_ROOT"
export TMPDIR="${TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${TMP_ROOT}/.cache"
export TORCH_HOME="${TMP_ROOT}/torch_cache"
export MPLCONFIGDIR="${TMP_ROOT}/matplotlib"
export WANDB_DIR="${TMP_ROOT}/wandb"
export HF_HOME="${TMP_ROOT}/hf"
export TRANSFORMERS_CACHE="${TMP_ROOT}/hf/transformers"
export HF_DATASETS_CACHE="${TMP_ROOT}/hf/datasets"
export PIP_CACHE_DIR="${TMP_ROOT}/pip-cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${ACTUAL_COMMIT}"
echo "config=${CONFIG}"
echo "config_sha256=${ACTUAL_CONFIG_SHA256}"
echo "mode=${MODE}"
echo "output_dir=${OUTPUT_DIR:-none}"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "slurm_tmpdir_writable=true"

cd "$REPO_ROOT"
python "$ENTRYPOINT" \
  --config "$CONFIG" \
  --preflight-only

if [ "$MODE" = "preflight" ]; then
  exit 0
fi
if [ "$MODE" = "smoke" ]; then
  python "$ENTRYPOINT" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 2 \
    --nn-batch-size 2 \
    --smoke
else
  python "$ENTRYPOINT" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 16 \
    --nn-batch-size 64
fi
