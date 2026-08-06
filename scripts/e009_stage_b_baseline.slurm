#!/bin/bash
#SBATCH --job-name=e009_b_base
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-b-baseline-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-b-baseline-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"
: "${E009_REMOTE_BRANCH_COMMIT:?E009_REMOTE_BRANCH_COMMIT is required}"
case "$E009_REPO_ROOT" in /*) ;; *) echo "error: E009_REPO_ROOT must be absolute" >&2; exit 1;; esac

REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
ENTRYPOINT="${REPO_ROOT}/experiments/09_stage_b_baseline_evaluation.py"
CONFIG="${REPO_ROOT}/configs/e009_stage_b_evaluation.json"
EXPECTED_CONFIG_SHA256="4a8dcbf46af72b87ca21113795cfa42991cf6bd5053213ed65789046ba56becf"
OUTPUT_DIR=/home/xggh8/data/zw-lab/e009_stage_b_baseline
MODE="${1:-}"

test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$E009_REPO_COMMIT" || { echo "error: repository commit mismatch" >&2; exit 1; }
test "$E009_REMOTE_BRANCH_COMMIT" = "$E009_REPO_COMMIT" || { echo "error: unpublished execution commit" >&2; exit 1; }
test -z "$(git -C "$REPO_ROOT" status --porcelain=v1)" || { echo "error: repository worktree is not clean" >&2; exit 1; }
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_CONFIG_SHA256" || { echo "error: evaluation config hash mismatch" >&2; exit 1; }

case "$MODE" in
  inventory)
    test ! -e "$OUTPUT_DIR" || { echo "error: output collision" >&2; exit 1; }
    MODE_ARGS=(--inventory-only)
    ;;
  summarize)
    test -f "$OUTPUT_DIR/checkpoint_pool_identity.json" || { echo "error: frozen pool missing" >&2; exit 1; }
    MODE_ARGS=(--summarize)
    ;;
  *) echo "error: usage: sbatch $0 {inventory|summarize}" >&2; exit 1 ;;
esac

TMP_ROOT="${SLURM_TMPDIR:-/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_stage_b_baseline_${SLURM_JOB_ID}}"
mkdir -p "$TMP_ROOT"
test -w "$TMP_ROOT" || { echo "error: temporary directory is not writable" >&2; exit 1; }
export TMPDIR="$TMP_ROOT/tmp" XDG_CACHE_HOME="$TMP_ROOT/.cache" TORCH_HOME="$TMP_ROOT/torch"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib" PYTHONPYCACHEPREFIX="$TMP_ROOT/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

echo "repository_commit=${E009_REPO_COMMIT}"
echo "remote_branch_commit=${E009_REMOTE_BRANCH_COMMIT}"
echo "mode=${MODE}"
echo "baseline_only=true"
echo "pilot_seeds=20000..20127"
echo "e008_swaps=false"
cd "$REPO_ROOT"
python "$ENTRYPOINT" --output-dir "$OUTPUT_DIR" "${MODE_ARGS[@]}"
