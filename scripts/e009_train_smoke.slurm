#!/bin/bash
#SBATCH --job-name=e009_smoke
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e009-smoke-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e009-smoke-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

: "${E009_REPO_ROOT:?E009_REPO_ROOT is required}"
exec "${E009_REPO_ROOT}/scripts/e009_training_entrypoint.sh" \
  configs/e009_smoke_edm2k_1kimg.yaml fresh
