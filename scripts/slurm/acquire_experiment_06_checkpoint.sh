#!/usr/bin/env bash
#SBATCH --job-name=e006-acquire
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

export SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID}}"
mkdir -p "$SLURM_TMPDIR"
export TMPDIR="$SLURM_TMPDIR/tmp"
export XDG_CACHE_HOME="$SLURM_TMPDIR/.cache"
export TORCH_HOME="$SLURM_TMPDIR/torch_cache"
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
export WANDB_DIR="$SLURM_TMPDIR/wandb"
export HF_HOME="$SLURM_TMPDIR/hf"
export TRANSFORMERS_CACHE="$SLURM_TMPDIR/hf/transformers"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf/datasets"
export PIP_CACHE_DIR="$SLURM_TMPDIR/pip-cache"

mkdir -p "$TMPDIR"

CHECKPOINT_NAME="256x256_diffusion_uncond.pt"
CHECKPOINT_URL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/${CHECKPOINT_NAME}"
EXPECTED_SIZE="2211383297"
EXPECTED_MD5="fd9dd2335b8736d521de0aed54bd90ca"
PERSISTENT_ROOT="${E006_MODEL_ROOT:-${HOME}/data/spectral-diffusion-playground/models}"
TEMP_CHECKPOINT="${SLURM_TMPDIR}/${CHECKPOINT_NAME}"

curl --fail --location --retry 4 --output "$TEMP_CHECKPOINT" "$CHECKPOINT_URL"

ACTUAL_SIZE="$(stat -c %s "$TEMP_CHECKPOINT")"
ACTUAL_MD5="$(md5sum "$TEMP_CHECKPOINT" | awk '{print $1}')"
ACTUAL_SHA256="$(sha256sum "$TEMP_CHECKPOINT" | awk '{print $1}')"

test "$ACTUAL_SIZE" = "$EXPECTED_SIZE"
test "$ACTUAL_MD5" = "$EXPECTED_MD5"

mkdir -p "$PERSISTENT_ROOT"
install -m 0444 "$TEMP_CHECKPOINT" "$PERSISTENT_ROOT/$CHECKPOINT_NAME"

cat > "$PERSISTENT_ROOT/experiment_06_checkpoint_identity.json" <<EOF
{
  "filename": "$CHECKPOINT_NAME",
  "size_bytes": $ACTUAL_SIZE,
  "md5": "$ACTUAL_MD5",
  "sha256": "$ACTUAL_SHA256",
  "source_url": "$CHECKPOINT_URL",
  "slurm_job_id": "$SLURM_JOB_ID"
}
EOF

printf 'Checkpoint acquired and verified: %s\n' "$PERSISTENT_ROOT/$CHECKPOINT_NAME"
printf 'SHA-256: %s\n' "$ACTUAL_SHA256"
