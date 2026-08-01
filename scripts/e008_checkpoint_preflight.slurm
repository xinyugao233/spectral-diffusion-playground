#!/bin/bash
#SBATCH --job-name=e008_preflight
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e008-preflight-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e008-preflight-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "error: SLURM_JOB_ID is unset" >&2
  exit 1
fi
if [ -z "${E008_REPO_ROOT:-}" ]; then
  echo "error: E008_REPO_ROOT is required" >&2
  exit 1
fi
if [ -z "${E008_REPO_COMMIT:-}" ]; then
  echo "error: E008_REPO_COMMIT is required" >&2
  exit 1
fi
case "$E008_REPO_ROOT" in
  /*) ;;
  *)
    echo "error: E008_REPO_ROOT must be absolute: ${E008_REPO_ROOT}" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$E008_REPO_ROOT" && pwd -P)"
ENTRYPOINT="${REPO_ROOT}/experiments/08_checkpoint_baseline_preflight.py"
CONFIG="${REPO_ROOT}/configs/e008_checkpoint_preflight.json"
EXPECTED_CONFIG_SHA256="64ae5b405ab51020eb897003f85161224508546324970f7a8e5deabe657286bd"
FULL_OUTPUT="/home/xggh8/data/zw-lab/e008_checkpoint_preflight"

if [ ! -d "${REPO_ROOT}/.git" ]; then
  echo "error: expected Git repository at ${REPO_ROOT}" >&2
  exit 1
fi
if [ ! -f "$ENTRYPOINT" ] || [ ! -f "$CONFIG" ]; then
  echo "error: E008 entrypoint or config missing under ${REPO_ROOT}" >&2
  exit 1
fi
ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$E008_REPO_COMMIT" ]; then
  echo "error: repository commit mismatch: ${ACTUAL_COMMIT} != ${E008_REPO_COMMIT}" >&2
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

MODE="${1:-}"
case "$MODE" in
  inventory)
    if [ -e "$FULL_OUTPUT" ]; then
      echo "error: inventory output already exists: ${FULL_OUTPUT}" >&2
      exit 1
    fi
    OUTPUT_DIR="$FULL_OUTPUT"
    MODE_ARGS=(--inventory-only)
    ;;
  smoke)
    if [ ! -f "${FULL_OUTPUT}/candidate_pool_manifest.json" ]; then
      echo "error: frozen candidate pool is missing" >&2
      exit 1
    fi
    OUTPUT_DIR="/home/xggh8/scratch/zw-lab/e008_checkpoint_preflight_smoke_${SLURM_JOB_ID}"
    if [ -e "$OUTPUT_DIR" ]; then
      echo "error: smoke output already exists: ${OUTPUT_DIR}" >&2
      exit 1
    fi
    MODE_ARGS=(--run-pilot --smoke)
    ;;
  full)
    if [ ! -f "${FULL_OUTPUT}/candidate_pool_manifest.json" ]; then
      echo "error: frozen candidate pool is missing" >&2
      exit 1
    fi
    OUTPUT_DIR="$FULL_OUTPUT"
    MODE_ARGS=(--run-pilot --resume)
    ;;
  summarize)
    OUTPUT_DIR="$FULL_OUTPUT"
    MODE_ARGS=(--summarize)
    ;;
  validate)
    OUTPUT_DIR="$FULL_OUTPUT"
    MODE_ARGS=(--validate-only)
    ;;
  *)
    echo "error: usage: sbatch $0 {inventory|smoke|full|summarize|validate}" >&2
    exit 1
    ;;
esac

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE="provided"
else
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e008_${SLURM_JOB_ID}"
  TMP_SOURCE="fallback"
fi
mkdir -p "$TMP_ROOT"
if [ ! -d "$TMP_ROOT" ] || [ ! -w "$TMP_ROOT" ]; then
  echo "error: E008 temporary directory is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
export SLURM_TMPDIR="$TMP_ROOT"
export TMPDIR="${TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${TMP_ROOT}/.cache"
export TORCH_HOME="${TMP_ROOT}/torch_cache"
export MPLCONFIGDIR="${TMP_ROOT}/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${ACTUAL_COMMIT}"
echo "config=${CONFIG}"
echo "config_sha256=${ACTUAL_CONFIG_SHA256}"
echo "mode=${MODE}"
echo "output_dir=${OUTPUT_DIR}"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "slurm_tmpdir_writable=true"
echo "baseline_only=true"
echo "swap_execution_available=false"

cd "$REPO_ROOT"
python "$ENTRYPOINT" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device auto \
  "${MODE_ARGS[@]}"
