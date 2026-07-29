# Experiment 5: Matched Clean-Room Models

## Status

**Matched clean-room model pair validated; E005 evaluation not yet
implemented.**

Experiment 5 is a paper-derived clean-room reimplementation. The original
executed paper checkpoints and CIFAR-10 1K subset were not recovered. The
models below must not be described as the paper's checkpoints or as an exact
numerical reproduction.

This amendment replaces the unavailable paper models with an internally
matched pair. The EDM-1K member already exists. The EDM-50K member has now
been trained once from the frozen configuration and validated as the matched
clean-room counterpart. This document records model provenance only; it
contains no E005 residual curves or scientific results.

## Clean-Room EDM-1K Anchor

The anchor is:

```text
/home/xggh8/data/zw-lab/
  exp_004_standard_edm_n1000_40000kimg_20260415/
  network-snapshot-040000.pkl
```

Identity:

```text
checkpoint size:        223159918 bytes
checkpoint SHA-256:     8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1
config SHA-256:         a2cd6b4f424f7c043aa64dd84b0a243c56e1e8df9766acfd67d9d9652214c2dd
run-manifest SHA-256:   c837fd0d77a3096deb3015d32dcfcd02eb372b8938e35f56b725ff1b44c6b7dc
NVlabs EDM commit:      008a4e5316c8e3bfe61a62f874bddba254295afb
wrapper SHA-256:        f5ca28db4a167ba5ee6c26bf9ae9cf4bf3215b919a86e7ad8e7f8ae0ad10c142
```

The wrapper is
`/home/xggh8/projects/zw-lab/src/zwlab_edm/train_subset_sigma.py`.
It is untracked in the `zw-lab` repository, so its content hash, rather than
that repository's commit, is the stable wrapper identity.

The EDM checkout had local changes when the anchor ran. The preparation
therefore freezes the relevant source-file hashes in addition to the base
commit:

| Source file | SHA-256 |
| --- | --- |
| `train.py` | `e562dacd2f403e4a9dfe8c857078bb506d8d6825cb88478869cc78b7c4587f05` |
| `training/dataset.py` | `fd4a37cdcca57563d2c20e7dc22da5b59bc7f15d1f53f87b902bf16c0917c05b` |
| `training/training_loop.py` | `9cac3720de1bd122a5fb735a133707fbe708daa454e00232390112311ee77391` |
| `training/networks.py` | `5db27dcd96674b95c72d5e6491b879cdc35e24039ada3411b4b46a28ed1fe284` |
| `training/loss.py` | `b26f0937bc72b18cc5f9869c86035ca9b872138cdfae7994237e9733951e4415` |

Architecture and optimization:

- `EDMPrecond` around a positional-embedding `SongUNet`/DDPM++ model;
- standard encoder and decoder, `model_channels=128`,
  `channel_mult=[2,2,2]`, and 55,732,739 parameters;
- unconditional RGB training with `label_dim=0`;
- standard `EDMLoss` with `P_mean=-1.2`, `P_std=1.2`, and
  `sigma_data=0.5`;
- Adam with learning rate `0.001`, betas `(0.9,0.999)`, and epsilon `1e-8`;
- batch and per-GPU batch size `64`;
- seed `0`, duration `40,000 kimg`, and float32 training;
- EMA half-life `500 kimg`;
- dropout `0.13`, no augmentation, no x-flip;
- progress tick `1,000 kimg`, snapshots every two ticks, and state dumps every
  four ticks.

CIFAR-10 PNG images are decoded as CHW uint8 and transformed at training time
with `float32(image) / 127.5 - 1`, with no crop, resize, standardization,
augmentation, or label conditioning.

This checkpoint completed on 2026-04-18, after the paper. It is the **E005
clean-room EDM-1K anchor**, not the paper's EDM-1K checkpoint.

## Clean-Room 1K Subset

The complete ordered manifest is
[`data/e005_cifar10_subset_1k_indices.txt`](../data/e005_cifar10_subset_1k_indices.txt).
It contains one canonical CIFAR-10 training index per line, in the order
presented to the dataset after selection:

```python
indices = np.arange(50000, dtype=np.int64)
np.random.RandomState(0).shuffle(indices)
indices = np.sort(indices[:1000])
```

Identity:

```text
subset seed:                    0
index count:                    1000
little-endian int64 SHA-256:    f97076ea6db59a96dc81a59d1b573bc8aaecdb8efa1e93c0d79928bfbf8a43f8
newline-text SHA-256:           33bb509c48144464a48d3b945cc44c14f880a1e6c6470c283dc0ed65e22b1f29
```

Class distribution in canonical CIFAR-10 label order:

| Class | Count |
| --- | ---: |
| airplane | 100 |
| automobile | 112 |
| bird | 94 |
| cat | 113 |
| deer | 89 |
| dog | 107 |
| frog | 99 |
| horse | 98 |
| ship | 90 |
| truck | 98 |

This is the **E005 clean-room 1K subset**. It is not the unrecovered paper
subset.

## CIFAR-10 Archive

Both clean-room models use:

```text
path:     /home/xggh8/datasets/edm/cifar10-32x32-train50k.zip
size:     164500138 bytes
SHA-256:  795cdc1444465ae4e19e25a0615d05ba0a0e83caa5db6b1b811deaf4c7910dfa
contents: 50000 ordered RGB uint8 PNG images; labels absent
```

The archive's image names run from `img00000000.png` through
`img00049999.png`. The wrapper trains unconditionally, so the absent labels do
not affect either clean-room model.

## Matched EDM-50K Configuration

The frozen config is
[`configs/e005_edm50k_matched_40000kimg.yaml`](../configs/e005_edm50k_matched_40000kimg.yaml),
SHA-256
`464576709477f0ff74e12bbd66b8ac8afcb19dfa6f4127add42e3ac0e4efd106`.

Relative to the anchor config, exactly two fields differ:

| Field | EDM-1K anchor | Matched EDM-50K |
| --- | --- | --- |
| `experiment.name` | `exp_004_standard_edm_n1000_40000kimg_20260415` | `e005_edm50k_matched_40000kimg` |
| `dataset.subset_size` | `1000` | `50000` |

The name difference isolates output paths. The subset-size difference is the
experimental variable. All remaining dataset, architecture, loss, optimizer,
duration, EMA, precision, seed, cadence, and preprocessing fields are
identical.

Expected persistent outputs:

```text
/home/xggh8/scratch/zw-lab/e005_edm50k_matched_40000kimg/
/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/
```

The selected checkpoint is:

```text
/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/
network-snapshot-040000.pkl
```

Identity:

```text
checkpoint size:          223159918 bytes
checkpoint SHA-256:       a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
training-state path:      /home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/training-state-040000.pt
training-state SHA-256:   4af61f228ea5ca0f25897ba180e3e8c5466628fecffa039e98d3505d0bbfbcf9
config_used SHA-256:      464576709477f0ff74e12bbd66b8ac8afcb19dfa6f4127add42e3ac0e4efd106
run-manifest SHA-256:     777dc54cfa65556d2b68c99363659d19f343c26863b93f8612da56ccaf2b9c84
training-options SHA-256: a5051275ea83f013ecbd60688b758849d66b6224882a463bc31f7db7278a4541
```

The complete provenance record is
`configs/e005_edm50k_matched_40000kimg_manifest.json`.

## Training Safety and Preflight

The launcher requests one GPU, eight CPUs, 48 GB host memory, and 48 hours.
The anchor used approximately 46.2 hours, peaked at 11.68 GB allocated GPU
memory during startup and 8.43 GB during steady training, and used less than
2 GB host memory. Runtime remains hardware-dependent.

The anchor ran on Hellbender node `g039`, configured with L40S GPUs. The
corrected launcher therefore requests exactly one `L40S` using
`--gres=gpu:L40S:1`. Hellbender rejected the otherwise portable `--gpus=1`
form because its configured `select/cons_res` plugin does not support that
request. The typed request passed `sbatch --test-only` without creating a job.

Hellbender's `gpu` partition has a hard `MaxTime=2-00:00:00`, so the launcher
retains the 48-hour request. A 60-hour request is not valid on this partition.

Job `15315328` passed the complete provenance preflight and then exited with
code `1:0` before the training wrapper ran. Its stdout ended at
`preflight=pass mode=fresh config_only=False`, stderr was empty, and no output
directory or checkpoint was created. The next launcher command was the silent
`test -n "${SLURM_TMPDIR:-}"`; Hellbender had not provided `SLURM_TMPDIR`.
This ordering confirms the missing temporary-directory variable as the
operational failure.

The launcher:

- refuses execution without `SLURM_JOB_ID`;
- requires `E005_REPO_ROOT` to name an absolute existing checkout;
- requires `E005_REPO_COMMIT` to identify the exact expected checkout commit;
- rejects missing tracked launch files, a non-Git path, a wrong commit, or a
  dirty worktree before checking or creating training outputs;
- performs no download;
- validates source, wrapper, config, and archive hashes before training;
- refuses `fresh` mode when either persistent output directory is nonempty;
- permits resume only through explicit `resume` mode and only when a nonempty
  `training-state-*.pt` exists;
- preserves a provided writable `SLURM_TMPDIR`;
- otherwise creates the deterministic job-specific fallback
  `/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e005_${SLURM_JOB_ID}`;
- creates the fallback only after provenance and output-collision preflight
  passes, and preserves it on failure for diagnosis;
- routes temporary files and caches through the resolved `SLURM_TMPDIR`;
- keeps stdout and stderr under `/home/xggh8/data/zw-lab/`;
- never writes into or overwrites the existing EDM-1K run.

The clean execution checkout is:

```text
/cluster/pixstor/zwggh-lab/xinyu/projects/spectral-diffusion-playground
```

The launcher remains portable: this path is supplied through
`E005_REPO_ROOT`, not embedded in the script. The successful fresh-run command
used the corrected checkout commit
`14a0e1a95c74ee83699d63734a632dbf9b05a0c8`:

```bash
E005_REPO_ROOT=/cluster/pixstor/zwggh-lab/xinyu/projects/spectral-diffusion-playground \
E005_REPO_COMMIT=14a0e1a95c74ee83699d63734a632dbf9b05a0c8 \
sbatch --export=ALL,E005_REPO_ROOT,E005_REPO_COMMIT \
  scripts/e005_train_edm50k_matched.slurm \
  configs/e005_edm50k_matched_40000kimg.yaml fresh
```

This command submitted Slurm job `15315560`. An interrupted run would have used
the same command with `resume` replacing `fresh`, but no resume was needed.
Training and E005 evaluation remain separate reviewed phases.

## EDM-50K Training Validation

Job `15315560` completed successfully:

```text
Slurm state:       COMPLETED
exit code:         0:0
submitted:         2026-07-26T02:42:10
started:           2026-07-26T02:42:27
completed:         2026-07-28T00:45:24
elapsed:           1-22:02:57
time limit:        2-00:00:00
node:              g039
mode:              fresh
```

The final `stats.jsonl` row reports:

```text
tick:                         40
kimg:                         40000.0
loss mean:                    0.17222712068216225
cpu memory:                   1.5840072631835938 GB
peak GPU memory:              8.430673599243164 GB
peak GPU memory reserved:     9.0390625 GB
seconds per kimg:             4.144377708435059
```

Training reached exactly `40,000 kimg` and exited normally. No traceback,
CUDA error, out-of-memory error, quota error, I/O error, or NaN was found in
the Slurm logs.

Nonfatal warnings recorded in stderr:

- PyTorch sampler `data_source` deprecation warning.
- DDP gradient-stride performance warning.
- `init_process_group` / `barrier` device-id warning.
- `destroy_process_group()` not called before process exit warning.

These warnings did not stop training and did not change the completed Slurm
state.

Complete persistent snapshot list:

```text
network-snapshot-000000.pkl
network-snapshot-002000.pkl
network-snapshot-004000.pkl
network-snapshot-006000.pkl
network-snapshot-008000.pkl
network-snapshot-010000.pkl
network-snapshot-012000.pkl
network-snapshot-014000.pkl
network-snapshot-016000.pkl
network-snapshot-018000.pkl
network-snapshot-020000.pkl
network-snapshot-022000.pkl
network-snapshot-024000.pkl
network-snapshot-026000.pkl
network-snapshot-028000.pkl
network-snapshot-030000.pkl
network-snapshot-032000.pkl
network-snapshot-034000.pkl
network-snapshot-036000.pkl
network-snapshot-038000.pkl
network-snapshot-040000.pkl
```

The final EMA checkpoint was loaded successfully in the smoke test below,
which is the readability check for `network-snapshot-040000.pkl`.

## Model-Pair Equivalence Check

The completed EDM-50K run is matched to the EDM-1K anchor in all frozen
settings except:

| Field | EDM-1K anchor | Matched EDM-50K |
| --- | --- | --- |
| `experiment.name` | `exp_004_standard_edm_n1000_40000kimg_20260415` | `e005_edm50k_matched_40000kimg` |
| `dataset.subset_size` | `1000` | `50000` |

The validated `training_options.json` confirms:

- dataset size `50000`;
- `use_labels=false`;
- seed `0`;
- `EDMPrecond`/`SongUNet` with `model_channels=128` and
  `channel_mult=[2,2,2]`;
- `EDMLoss` with `P_mean=-1.2`, `P_std=1.2`, and `sigma_data=0.5`;
- Adam optimizer with learning rate `0.001`, betas `(0.9,0.999)`, and
  epsilon `1e-8`;
- batch size and per-GPU batch size `64`;
- duration `40000 kimg`;
- EMA half-life `500 kimg`;
- dropout `0.13`;
- `use_fp16=false`;
- no augmentation and no x-flip.

Resume-mode preflight after completion passed and revalidated the repository
commit, frozen config hash, EDM source-file hashes, wrapper hash, CIFAR-10
archive hash, and existing final training state.

## Checkpoint Smoke Test

The final EMA checkpoint was loaded in Slurm job `15425345`, a CPU-only
validation job on node `c043`. The smoke test used one CIFAR-10 training image,
`class_labels=None`, and sigma values `0.5` and `5.0`.

Output record:

```text
/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/e005_checkpoint_smoke_cpu.json
SHA-256: 29551a7e7387a795e9c76d3854517f00e3ae80851743d298e974ab4b4661ec6a
```

The smoke test confirmed:

- checkpoint deserialization succeeds;
- EMA network invocation succeeds with `class_labels=None`;
- inputs and outputs use NCHW layout;
- outputs are `torch.float32`;
- outputs are finite at both sigma values;
- no clamping or quantization is applied by the smoke script.

The smoke test is not an E005 evaluation. It produces no residual-energy
curves, no train/test comparison, and no scientific conclusion.

One earlier GPU smoke-test job, `15425299`, remained pending on scheduler
priority during validation. It was not canceled because cancellation was not
authorized in this phase.
