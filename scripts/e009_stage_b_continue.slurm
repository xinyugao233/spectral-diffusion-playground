#!/bin/bash
#SBATCH --job-name=e009_b_continue
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-b-continue-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-b-continue-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-06:00:00

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
TMP_ROOT="${SLURM_TMPDIR:-/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_stage_b_continue_${SLURM_JOB_ID}}"
mkdir -p "$TMP_ROOT"
test -d "$TMP_ROOT" && test -w "$TMP_ROOT"
export SLURM_TMPDIR="$(cd "$TMP_ROOT" && pwd -P)"
export TMPDIR="${SLURM_TMPDIR}/tmp"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/.cache"
export TORCH_HOME="${SLURM_TMPDIR}/torch_cache"
export MPLCONFIGDIR="${SLURM_TMPDIR}/matplotlib"
export WANDB_DIR="${SLURM_TMPDIR}/wandb"
export HF_HOME="${SLURM_TMPDIR}/hf"
export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf/transformers"
export HF_DATASETS_CACHE="${SLURM_TMPDIR}/hf/datasets"
export PIP_CACHE_DIR="${SLURM_TMPDIR}/pip-cache"
export PYTHONPYCACHEPREFIX="${SLURM_TMPDIR}/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/projects/zw-lab/src:/home/xggh8/projects/zw-lab:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

STAGE_A_ROOT=/home/xggh8/data/zw-lab/e009_edm5k_12000kimg
STAGE_A_BEFORE="${SLURM_TMPDIR}/stage_a_identity_before.json"
OUTPUT_DIR=/home/xggh8/data/zw-lab/e009_stage_b_edm5k_30000kimg

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E009_REPO_COMMIT}"
echo "validated_parent_implementation=9e5782f09a3e024c298dc5ce8da1c0f44c9b4fbd"
echo "resume_kind=extended_stage_b_state"
echo "start_kimg=13000"
echo "stop_kimg=30000"
echo "expected_new_checkpoints=14000..30000"
echo "evaluation=false"
echo "e008_swaps=false"

cd "$REPO_ROOT"
python scripts/e009_stage_b_preflight.py \
  --repo-root "$REPO_ROOT" \
  --expected-repo-commit "$E009_REPO_COMMIT" \
  --config "$CONFIG" \
  --mode continue \
  --stage-a-manifest-output "$STAGE_A_BEFORE"
python experiments/09_stage_b_warm_start.py --config "$CONFIG"
python scripts/e009_validate_stage_b_continuation.py \
  --output-dir "$OUTPUT_DIR" \
  --stage-a-root "$STAGE_A_ROOT" \
  --stage-a-before "$STAGE_A_BEFORE" \
  --edm-root /home/xggh8/edm
echo "stage_b_continuation=pass"
echo "evaluation_submitted=false"
echo "e008_swaps_submitted=false"
