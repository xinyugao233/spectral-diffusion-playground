#!/bin/bash
#SBATCH --job-name=e009_b_state_load
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-b-state-load-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-b-state-load-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"
case "$E009_REPO_ROOT" in
  /*) ;;
  *) echo "error: E009_REPO_ROOT must be absolute" >&2; exit 1 ;;
esac
if [ ! -d "$E009_REPO_ROOT/.git" ]; then
  echo "error: E009_REPO_ROOT is not the expected Git checkout" >&2
  exit 1
fi
REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
CONFIG="${REPO_ROOT}/configs/e009_stage_b_edm5k_30000kimg.yaml"
TMP_ROOT="${SLURM_TMPDIR:-/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_stage_b_state_load_${SLURM_JOB_ID}}"
mkdir -p "$TMP_ROOT"
test -d "$TMP_ROOT" && test -w "$TMP_ROOT"
export SLURM_TMPDIR="$(cd "$TMP_ROOT" && pwd -P)"
export TMPDIR="${SLURM_TMPDIR}/tmp"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/.cache"
export TORCH_HOME="${SLURM_TMPDIR}/torch_cache"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTHONPYCACHEPREFIX"
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/projects/zw-lab/src:/home/xggh8/projects/zw-lab:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

STAGE_A_ROOT=/home/xggh8/data/zw-lab/e009_edm5k_12000kimg
STAGE_A_BEFORE="${SLURM_TMPDIR}/stage_a_identity_before.json"
STATE=/home/xggh8/data/zw-lab/e009_stage_b_edm5k_30000kimg/training-state-013000.pt
SNAPSHOT=/home/xggh8/data/zw-lab/e009_stage_b_edm5k_30000kimg/network-snapshot-013000.pkl
OUTPUT=/home/xggh8/data/zw-lab/e009_stage_b_state_load_preflight_${SLURM_JOB_ID}.json

cd "$REPO_ROOT"
python scripts/e009_stage_b_preflight.py \
  --repo-root "$REPO_ROOT" \
  --expected-repo-commit "$E009_REPO_COMMIT" \
  --config "$CONFIG" \
  --mode state-load \
  --stage-a-manifest-output "$STAGE_A_BEFORE"
python scripts/e009_stage_b_state_load_preflight.py \
  --state "$STATE" \
  --state-sha256 8bb1aabceee959ce2478a108b27ad6b34313cf8329cba2b048c9446077a7a130 \
  --snapshot "$SNAPSHOT" \
  --snapshot-sha256 6d181c0102e93cfe1c43005675e7c76e01fae18afd402337a35ebc8b2128371c \
  --stage-a-root "$STAGE_A_ROOT" \
  --stage-a-before "$STAGE_A_BEFORE" \
  --edm-root /home/xggh8/edm \
  --output "$OUTPUT"
echo "state_load_only_preflight=pass"
echo "optimizer_steps_taken=0"
