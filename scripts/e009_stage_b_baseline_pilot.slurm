#!/bin/bash
#SBATCH --job-name=e009_b_pilot
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-b-pilot-%A_%a.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-b-pilot-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"
: "${E009_REMOTE_BRANCH_COMMIT:?E009_REMOTE_BRANCH_COMMIT is required}"
ROLES=(edm_5k edm_1k)
ROLE="${ROLES[$SLURM_ARRAY_TASK_ID]}"
REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
CONFIG="${REPO_ROOT}/configs/e009_stage_b_evaluation.json"
OUTPUT=/home/xggh8/data/zw-lab/e009_stage_b_baseline
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$E009_REPO_COMMIT" || exit 1
test "$E009_REMOTE_BRANCH_COMMIT" = "$E009_REPO_COMMIT" || exit 1
test -z "$(git -C "$REPO_ROOT" status --porcelain=v1)" || exit 1
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = 4a8dcbf46af72b87ca21113795cfa42991cf6bd5053213ed65789046ba56becf || exit 1
test -f "$OUTPUT/shards/$ROLE/candidate_pool_manifest.json" || exit 1
TMP_ROOT="${SLURM_TMPDIR:-/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_stage_b_pilot_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TMP_ROOT"
export TMPDIR="$TMP_ROOT/tmp" XDG_CACHE_HOME="$TMP_ROOT/.cache" TORCH_HOME="$TMP_ROOT/torch"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib" PYTHONPYCACHEPREFIX="$TMP_ROOT/pycache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
echo "role=${ROLE}"
echo "pilot_seeds=20000..20127"
echo "baseline_only=true"
echo "e008_swaps=false"
cd "$REPO_ROOT"
python experiments/09_stage_b_baseline_evaluation.py --run-pilot --role "$ROLE" --output-dir "$OUTPUT"
