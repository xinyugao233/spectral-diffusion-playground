#!/bin/bash
#SBATCH --job-name=e009_baseline
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-baseline-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-baseline-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"
case "$E009_REPO_ROOT" in /*) ;; *) echo "error: E009_REPO_ROOT must be absolute" >&2; exit 1;; esac

REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
ENTRYPOINT="${REPO_ROOT}/experiments/09_stage_a_baseline_evaluation.py"
CONFIG="${REPO_ROOT}/configs/e009_stage_a_evaluation.json"
EXPECTED_CONFIG_SHA256="6eb4edcae68d3533075401146082fdf7edb0dc40d2fc88bfd82722a342aebd32"
OUTPUT_DIR="/home/xggh8/data/zw-lab/e009_stage_a_baseline"
MODE="${1:-}"

test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$E009_REPO_COMMIT" || {
  echo "error: repository commit mismatch" >&2; exit 1;
}
test -z "$(git -C "$REPO_ROOT" status --porcelain=v1)" || {
  echo "error: repository worktree is not clean" >&2; exit 1;
}
test -f "$ENTRYPOINT" && test -f "$CONFIG" || {
  echo "error: E009 evaluator or config is missing" >&2; exit 1;
}
ACTUAL_CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
test "$ACTUAL_CONFIG_SHA256" = "$EXPECTED_CONFIG_SHA256" || {
  echo "error: E009 evaluation config hash mismatch" >&2; exit 1;
}

case "$MODE" in
  inventory)
    test ! -e "$OUTPUT_DIR" || { echo "error: output collision" >&2; exit 1; }
    MODE_ARGS=(--inventory-only)
    ;;
  summarize)
    MODE_ARGS=(--summarize)
    ;;
  *)
    echo "error: usage: sbatch $0 {inventory|summarize}" >&2
    exit 1
    ;;
esac

TMP_ROOT="${SLURM_TMPDIR:-/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_eval_${SLURM_JOB_ID}}"
mkdir -p "$TMP_ROOT"
test -w "$TMP_ROOT" || { echo "error: temporary directory is not writable" >&2; exit 1; }
export TMPDIR="$TMP_ROOT/tmp"
export XDG_CACHE_HOME="$TMP_ROOT/.cache"
export TORCH_HOME="$TMP_ROOT/torch"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$TMP_ROOT/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E009_REPO_COMMIT}"
echo "config_sha256=${ACTUAL_CONFIG_SHA256}"
echo "mode=${MODE}"
echo "output_dir=${OUTPUT_DIR}"
echo "baseline_only=true"
echo "pilot_seeds=20000..20127"
echo "swap_execution_available=false"

cd "$REPO_ROOT"
python "$ENTRYPOINT" --output-dir "$OUTPUT_DIR" "${MODE_ARGS[@]}"
