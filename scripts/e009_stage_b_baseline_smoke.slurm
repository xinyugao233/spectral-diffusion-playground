#!/bin/bash
#SBATCH --job-name=e009_b_base_smoke
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-b-baseline-smoke-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-b-baseline-smoke-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"
: "${E009_REMOTE_BRANCH_COMMIT:?E009_REMOTE_BRANCH_COMMIT is required}"
REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
CONFIG="${REPO_ROOT}/configs/e009_stage_b_evaluation.json"
POOL=/home/xggh8/data/zw-lab/e009_stage_b_baseline
OUTPUT=/home/xggh8/data/zw-lab/e009_stage_b_baseline_smoke_${SLURM_JOB_ID}
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$E009_REPO_COMMIT" || exit 1
test "$E009_REMOTE_BRANCH_COMMIT" = "$E009_REPO_COMMIT" || exit 1
test -z "$(git -C "$REPO_ROOT" status --porcelain=v1)" || exit 1
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = 4a8dcbf46af72b87ca21113795cfa42991cf6bd5053213ed65789046ba56becf || exit 1
test -f "$POOL/checkpoint_pool_identity.json" || exit 1
test ! -e "$OUTPUT" || exit 1
TMP_ROOT="${SLURM_TMPDIR:-/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_stage_b_baseline_smoke_${SLURM_JOB_ID}}"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT/tmp" XDG_CACHE_HOME="$TMP_ROOT/.cache" TORCH_HOME="$TMP_ROOT/torch"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib" PYTHONPYCACHEPREFIX="$TMP_ROOT/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
echo "smoke_checkpoints=edm_5k:30000,edm_1k:12000"
echo "smoke_seeds=20000,20001"
echo "baseline_only=true"
echo "e008_swaps=false"
cd "$REPO_ROOT"
python experiments/09_stage_b_baseline_evaluation.py --smoke --output-dir "$OUTPUT"
