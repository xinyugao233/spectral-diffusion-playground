# Experiment 6 Execution Log

This log records infrastructure and scheduler decisions separately from
scientific results. Experiment 6 is not complete until the frozen numerical,
repeatability, uncertainty, and failure-analysis gates pass.

## Source Identity

- Playground execution commit:
  `fd48730e9cc159d65d0af025659dae20bb36602b`
- Guided-diffusion commit:
  `22e0df8183507e13a7813f8d38d51b072ca1e67c`
- Execution root:
  `/cluster/pixstor/zwggh-lab/xinyu/experiments/playground_e006_fixed_model`

The lab-backed root was used because cloning into the Hellbender home-backed
`~/projects` path failed with `Disk quota exceeded`. No existing home files
were deleted.

## Checkpoint Acquisition

Initial acquisition job `15305876` failed before download because Hellbender
did not define `SLURM_TMPDIR`. Commit `7519542` added a node-local `/tmp`
fallback while preserving all cache-routing rules.

Acquisition job `15305877` then completed successfully:

- filename: `256x256_diffusion_uncond.pt`
- size: `2,211,383,297` bytes
- MD5: `fd9dd2335b8736d521de0aed54bd90ca`
- SHA-256:
  `a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1`

## Preflight

Slurm smoke-check job `15305890` completed at playground commit `fd48730`.
It compiled repository Python files and imported the frozen E006 module without
loading the dataset or checkpoint.

## Evaluation Submission

Evaluation job: `15305891`.

The job was initially submitted to the `gpu` partition with one A100 and an
eight-hour walltime. Scheduler inspection showed an idle L40S node while the
A100-constrained job had a roughly four-hour priority delay. The existing
pending job was updated, rather than duplicated:

- partition: `requeue`
- GPU: one L40S
- walltime: 30 minutes

These are scheduling and hardware changes only. The source commit, checkpoint,
dataset, model flags, timesteps, seeds, cutoffs, batch size, two-pass
repeatability gate, metrics, and output schema are unchanged. The run manifest
must record the actual GPU. Slurm accounting remains the source for partition
and walltime.

Current status at 2026-07-25 06:25 CDT: pending for priority. No raw E006 scores
or figures exist yet.
