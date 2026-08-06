#!/bin/bash
#SBATCH --job-name=e009_b_resume_smoke
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-b-smoke-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-b-smoke-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

set -euo pipefail

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
: "${E009_REPO_COMMIT:?E009_REPO_COMMIT is required}"

case "$E009_REPO_ROOT" in
  /*) ;;
  *) echo "error: E009_REPO_ROOT must be absolute" >&2; exit 1 ;;
esac
REPO_ROOT="$(cd "$E009_REPO_ROOT" && pwd -P)"
CONFIG="${REPO_ROOT}/configs/e009_stage_b_edm5k_13000kimg_smoke.yaml"
PREFLIGHT="${REPO_ROOT}/scripts/e009_stage_b_preflight.py"
TRAINER="${REPO_ROOT}/experiments/09_stage_b_warm_start.py"
VALIDATOR="${REPO_ROOT}/scripts/e009_validate_stage_b_smoke.py"
for path in "$CONFIG" "$PREFLIGHT" "$TRAINER" "$VALIDATOR" "${REPO_ROOT}/.git"; do
  if [ ! -e "$path" ]; then
    echo "error: required path is missing: ${path}" >&2
    exit 1
  fi
done

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE=provided
else
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e009_stage_b_smoke_${SLURM_JOB_ID}"
  TMP_SOURCE=fallback
  mkdir -p "$TMP_ROOT"
fi
if [ ! -d "$TMP_ROOT" ] || [ ! -w "$TMP_ROOT" ]; then
  echo "error: temporary directory is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
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

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/projects/zw-lab/src:/home/xggh8/projects/zw-lab:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

STAGE_A_ROOT=/home/xggh8/data/zw-lab/e009_edm5k_12000kimg
STAGE_A_BEFORE="${SLURM_TMPDIR}/stage_a_identity_before.json"
SCRATCH_DIR=/cluster/pixstor/zwggh-lab/xinyu/e009_training_scratch/e009_stage_b_edm5k_30000kimg
FINAL_DIR=/home/xggh8/data/zw-lab/e009_stage_b_edm5k_30000kimg

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E009_REPO_COMMIT}"
echo "config=${CONFIG}"
echo "warm_start=true"
echo "exact_stage_a_continuation=false"
echo "stage_b_seed=1"
echo "start_kimg=12000"
echo "stop_kimg=13000"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "output_dir=${FINAL_DIR}"
echo "full_stage_b_continuation=false"
echo "evaluation=false"
echo "e008_swaps=false"

cd "$REPO_ROOT"
python "$PREFLIGHT" \
  --repo-root "$REPO_ROOT" \
  --expected-repo-commit "$E009_REPO_COMMIT" \
  --config "$CONFIG" \
  --stage-a-manifest-output "$STAGE_A_BEFORE"

python "$TRAINER" --config "$CONFIG"

python "$VALIDATOR" \
  --scratch-dir "$SCRATCH_DIR" \
  --final-dir "$FINAL_DIR" \
  --stage-a-root "$STAGE_A_ROOT" \
  --stage-a-before "$STAGE_A_BEFORE" \
  --edm-root /home/xggh8/edm \
  --expected-parent-state-sha256 \
  1073e68c9f45123b53811a12a56a565f296a5ab846212d22e5027bbd81d685f5 \
  --expected-parent-ema-sha256 \
  a77c19588f9a4f877de961102c16901ee07bbd87e0e4ace6164f92f40c406d58 \
  --expected-start-kimg 12000 \
  --expected-final-kimg 13000

echo "stage_b_resume_smoke=pass"
echo "full_stage_b_continuation_submitted=false"
