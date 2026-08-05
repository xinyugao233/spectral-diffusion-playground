#!/bin/bash
#SBATCH --job-name=e009_subsets
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-subsets-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-subsets-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"

case "$E009_REPO_ROOT" in
  /*) ;;
  *) echo "error: E009_REPO_ROOT must be absolute" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
if [ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$E009_REPO_COMMIT" ]; then
  echo "error: repository commit mismatch" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]; then
  echo "error: repository worktree is not clean" >&2
  exit 1
fi

OUTPUT_DIR="/home/xggh8/data/zw-lab/e009_stage_a_subsets"
if [ -e "$OUTPUT_DIR" ]; then
  echo "error: subset output already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_subsets_${SLURM_JOB_ID}"
mkdir -p "$TMP_ROOT"
export SLURM_TMPDIR="$TMP_ROOT"
export TMPDIR="$TMP_ROOT/tmp"
export XDG_CACHE_HOME="$TMP_ROOT/.cache"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TMP_ROOT/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E009_REPO_COMMIT}"
echo "output_dir=${OUTPUT_DIR}"
echo "mode=subset_preparation"

PYTHONPATH="${REPO_ROOT}/src" python \
  "${REPO_ROOT}/scripts/e009_build_nested_subsets.py" \
  --label-root /home/xggh8/datasets/edm/cifar-10-batches-py \
  --source-archive /home/xggh8/datasets/edm/cifar10-32x32-train50k.zip \
  --anchor-indices "${REPO_ROOT}/data/e005_cifar10_subset_1k_indices.txt" \
  --output-dir "$OUTPUT_DIR" \
  --sizes 2000 5000 10000 \
  --expansion-seed 9
