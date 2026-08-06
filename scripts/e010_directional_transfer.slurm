#!/bin/bash
#SBATCH --job-name=e010_transfer
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e010-transfer-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e010-transfer-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "error: SLURM_JOB_ID is unset" >&2
  exit 1
fi
if [ -z "${E010_REPO_ROOT:-}" ] || [ -z "${E010_REPO_COMMIT:-}" ]; then
  echo "error: E010_REPO_ROOT and E010_REPO_COMMIT are required" >&2
  exit 1
fi
case "$E010_REPO_ROOT" in
  /*) ;;
  *) echo "error: E010_REPO_ROOT must be absolute" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$E010_REPO_ROOT" && pwd -P)"
ENTRYPOINT="${REPO_ROOT}/experiments/10_directional_memorization_transfer.py"
CONFIG="${REPO_ROOT}/configs/e010_directional_memorization_transfer.json"
if [ ! -d "${REPO_ROOT}/.git" ] || [ ! -f "$ENTRYPOINT" ] || [ ! -f "$CONFIG" ]; then
  echo "error: E010 checkout is incomplete: ${REPO_ROOT}" >&2
  exit 1
fi
if [ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$E010_REPO_COMMIT" ]; then
  echo "error: E010 repository commit mismatch" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]; then
  echo "error: E010 repository worktree is not clean" >&2
  exit 1
fi

MODE="${1:-}"
case "$MODE" in
  preflight) OUTPUT_DIR="" ;;
  smoke) OUTPUT_DIR="/home/xggh8/scratch/zw-lab/e010_directional_transfer_smoke_${SLURM_JOB_ID}" ;;
  full) OUTPUT_DIR="/home/xggh8/data/zw-lab/e010_directional_memorization_transfer" ;;
  resume) OUTPUT_DIR="/home/xggh8/data/zw-lab/e010_directional_memorization_transfer" ;;
  *) echo "error: usage: sbatch $0 {preflight|smoke|full|resume}" >&2; exit 1 ;;
esac
if [ -n "$OUTPUT_DIR" ] && [ -e "$OUTPUT_DIR" ] && [ "$MODE" != "resume" ]; then
  echo "error: output path already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE="provided"
else
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e010_${SLURM_JOB_ID}"
  TMP_SOURCE="fallback"
fi
mkdir -p "$TMP_ROOT"
if [ ! -d "$TMP_ROOT" ] || [ ! -w "$TMP_ROOT" ]; then
  echo "error: temporary directory is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
export SLURM_TMPDIR="$TMP_ROOT"
export TMPDIR="${TMP_ROOT}/tmp"
export MPLCONFIGDIR="${TMP_ROOT}/matplotlib"
export XDG_CACHE_HOME="${TMP_ROOT}/cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E010_REPO_COMMIT}"
echo "mode=${MODE}"
echo "output_dir=${OUTPUT_DIR:-none}"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "training_executed=false"

cd "$REPO_ROOT"
if [ "$MODE" = "preflight" ]; then
  python "$ENTRYPOINT" --mode preflight --config "$CONFIG"
elif [ "$MODE" = "resume" ]; then
  python "$ENTRYPOINT" --mode full --resume --config "$CONFIG" --output-dir "$OUTPUT_DIR"
else
  python "$ENTRYPOINT" --mode "$MODE" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
fi
