#!/bin/bash
#SBATCH --job-name=e005_edm50k
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e005-edm50k-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e005-edm50k-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
case "$E005_REPO_ROOT" in
  /*) ;;
  *)
    echo "E005_REPO_ROOT must be absolute: ${E005_REPO_ROOT}" >&2
    exit 1
    ;;
esac

if [ -z "${E005_REPO_COMMIT:-}" ]; then
  echo "E005_REPO_COMMIT is required" >&2
  exit 1
fi

LAUNCHER_DRY_RUN=false
if [ "$#" -eq 3 ] && [ "$3" = "--launcher-dry-run" ]; then
  LAUNCHER_DRY_RUN=true
elif [ "$#" -ne 2 ]; then
  echo "Usage: sbatch scripts/e005_train_edm50k_matched.slurm CONFIG {fresh|resume} [--launcher-dry-run]" >&2
  exit 1
fi

if [ ! -d "$E005_REPO_ROOT" ]; then
  echo "E005_REPO_ROOT does not exist: ${E005_REPO_ROOT}" >&2
  exit 1
fi
REPO_ROOT="$(cd "$E005_REPO_ROOT" && pwd -P)"
PREFLIGHT="${REPO_ROOT}/scripts/e005_preflight.py"
EXPECTED_CONFIG="${REPO_ROOT}/configs/e005_edm50k_matched_40000kimg.yaml"
MANIFEST="${REPO_ROOT}/configs/e005_edm50k_matched_40000kimg_manifest.json"

for required_path in "$PREFLIGHT" "$EXPECTED_CONFIG" "$MANIFEST" "${REPO_ROOT}/.git"; do
  if [ ! -e "$required_path" ]; then
    echo "Required repository path is missing: ${required_path}" >&2
    exit 1
  fi
done

case "$1" in
  /*) CONFIG="$1" ;;
  *) CONFIG="${REPO_ROOT}/$1" ;;
esac
if [ "$CONFIG" != "$EXPECTED_CONFIG" ]; then
  echo "Config must be the frozen file under E005_REPO_ROOT: ${EXPECTED_CONFIG}" >&2
  exit 1
fi

MODE="$2"
WRAPPER="/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py"

if [ "$MODE" != "fresh" ] && [ "$MODE" != "resume" ]; then
  echo "Mode must be fresh or resume" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit_expected=${E005_REPO_COMMIT}"
echo "training_mode=${MODE}"
echo "config=${CONFIG}"
echo "manifest=${MANIFEST}"
echo "expected_output=/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg"

python "$PREFLIGHT" \
  --repo-root "$REPO_ROOT" \
  --expected-repo-commit "$E005_REPO_COMMIT" \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --mode "$MODE"

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  SLURM_TMPDIR_SOURCE=provided
  if [ ! -d "$TMP_ROOT" ]; then
    echo "error: provided SLURM_TMPDIR is not a directory: ${TMP_ROOT}" >&2
    exit 1
  fi
else
  FALLBACK_TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp"
  if [ "$LAUNCHER_DRY_RUN" = true ] && [ -n "${E005_DRY_RUN_TMP_ROOT:-}" ]; then
    FALLBACK_TMP_ROOT="$E005_DRY_RUN_TMP_ROOT"
  fi
  case "$FALLBACK_TMP_ROOT" in
    /*) ;;
    *)
      echo "error: fallback temporary root must be absolute: ${FALLBACK_TMP_ROOT}" >&2
      exit 1
      ;;
  esac
  TMP_ROOT="${FALLBACK_TMP_ROOT}/e005_${SLURM_JOB_ID}"
  SLURM_TMPDIR_SOURCE=fallback
  if ! mkdir -p "$TMP_ROOT"; then
    echo "error: failed to create fallback SLURM_TMPDIR: ${TMP_ROOT}" >&2
    exit 1
  fi
fi

if [ ! -w "$TMP_ROOT" ]; then
  echo "error: SLURM_TMPDIR is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
TMP_ROOT="$(cd "$TMP_ROOT" && pwd -P)"
export SLURM_TMPDIR="$TMP_ROOT"

echo "slurm_tmpdir_source=${SLURM_TMPDIR_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "slurm_tmpdir_writable=true"
echo "output_dir=/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg"
echo "mode=${MODE}"

export TMPDIR="${TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${TMP_ROOT}/.cache"
export TORCH_HOME="${TMP_ROOT}/torch_cache"
export MPLCONFIGDIR="${TMP_ROOT}/matplotlib"
export WANDB_DIR="${TMP_ROOT}/wandb"
export HF_HOME="${TMP_ROOT}/hf"
export TRANSFORMERS_CACHE="${TMP_ROOT}/hf/transformers"
export HF_DATASETS_CACHE="${TMP_ROOT}/hf/datasets"
export PIP_CACHE_DIR="${TMP_ROOT}/pip-cache"
export PYTHONPYCACHEPREFIX="${TMP_ROOT}/pycache"
mkdir -p \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX"

if [ "$LAUNCHER_DRY_RUN" = true ]; then
  echo "launcher_dry_run=pass"
  exit 0
fi

export PYTHONPATH="/home/xggh8/projects/zw-lab/src:/home/xggh8/projects/zw-lab:${PYTHONPATH:-}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

python "$WRAPPER" --config "$CONFIG"
