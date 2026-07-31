#!/bin/bash
#SBATCH --job-name=e005_eval
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e005-eval-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e005-eval-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "error: SLURM_JOB_ID is unset" >&2
  exit 1
fi
if [ -z "${E005_REPO_ROOT:-}" ]; then
  echo "E005_REPO_ROOT is required" >&2
  exit 1
fi
if [ -z "${E005_REPO_COMMIT:-}" ]; then
  echo "E005_REPO_COMMIT is required" >&2
  exit 1
fi
case "$E005_REPO_ROOT" in
  /*) ;;
  *)
    echo "E005_REPO_ROOT must be absolute: ${E005_REPO_ROOT}" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$E005_REPO_ROOT" && pwd -P)"
EXPECTED_SCRIPT="${REPO_ROOT}/experiments/05_spectral_residual_curves.py"
if [ ! -f "$EXPECTED_SCRIPT" ]; then
  echo "Required evaluator missing: ${EXPECTED_SCRIPT}" >&2
  exit 1
fi
if [ ! -d "${REPO_ROOT}/.git" ]; then
  echo "Expected Git repository at ${REPO_ROOT}" >&2
  exit 1
fi
ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [ "$ACTUAL_COMMIT" != "$E005_REPO_COMMIT" ]; then
  echo "Repository commit mismatch: ${ACTUAL_COMMIT} != ${E005_REPO_COMMIT}" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]; then
  echo "Repository worktree is not clean" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 1
fi

OUTPUT_DIR="${1:-/home/xggh8/data/zw-lab/e005_spectral_residual_curves}"
MODE="${2:-full}"
if [ "$MODE" != "full" ] && [ "$MODE" != "smoke" ]; then
  echo "Usage: sbatch $0 [OUTPUT_DIR] {full|smoke}" >&2
  exit 1
fi
if [ -e "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Output directory is nonempty: ${OUTPUT_DIR}" >&2
  exit 1
fi

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE="provided"
else
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e005_eval_${SLURM_JOB_ID}"
  TMP_SOURCE="fallback"
fi
mkdir -p "$TMP_ROOT"
test -d "$TMP_ROOT"
test -w "$TMP_ROOT"
export SLURM_TMPDIR="$TMP_ROOT"
export TMPDIR="${TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${TMP_ROOT}/.cache"
export TORCH_HOME="${TMP_ROOT}/torch_cache"
export MPLCONFIGDIR="${TMP_ROOT}/matplotlib"
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${ACTUAL_COMMIT}"
echo "output_dir=${OUTPUT_DIR}"
echo "mode=${MODE}"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "slurm_tmpdir_writable=true"

cd "$REPO_ROOT"
if [ "$MODE" = "smoke" ]; then
  python experiments/05_spectral_residual_curves.py \
    --output-dir "$OUTPUT_DIR" \
    --models edm_1k edm_50k \
    --splits train test \
    --images-per-split 2 \
    --noise-repeats 1 \
    --batch-size 4 \
    --bootstrap-resamples 32 \
    --smoke
else
  python experiments/05_spectral_residual_curves.py \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 128
fi
