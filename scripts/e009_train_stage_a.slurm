#!/bin/bash
#SBATCH --job-name=e009_stage_a
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-stage-a-%A_%a.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-stage-a-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00

set -euo pipefail

: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
CONFIGS=(
  configs/e009_edm2k_12000kimg.yaml
  configs/e009_edm5k_12000kimg.yaml
  configs/e009_edm10k_12000kimg.yaml
)
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ] || [ "$SLURM_ARRAY_TASK_ID" -gt 2 ]; then
  echo "error: expected SLURM_ARRAY_TASK_ID in 0..2" >&2
  exit 1
fi

exec bash "${E009_REPO_ROOT}/scripts/e009_training_entrypoint.sh" \
  "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" fresh
