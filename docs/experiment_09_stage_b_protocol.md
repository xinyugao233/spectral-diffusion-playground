# E009 Stage B: Matched-Exposure 5K Warm-Start Extension

Status: **FULL CONTINUATION COMPLETE — BASELINE EVALUATION PENDING**

Date frozen: 2026-08-05

> Stage B warm-starts from the verified 5K training network and optimizer at
> 12K, loads the separately frozen 12K EMA, and trains to at most 30K kimg
> under a newly frozen RNG seed. It does not execute E008 swaps or use
> confirmatory seeds.

## Pre-Execution Amendment: Exact Resume Is Unavailable

Source inspection before any Stage B execution established that the Stage A
state stores only `net` and `optimizer_state`. The 12K EMA is stored in a
separate network snapshot; NumPy, Torch, CUDA, sampler, and DataLoader RNG
states were not serialized; and progress is reconstructed from the artifact
filename. Exact continuation of the uninterrupted Stage A stochastic
trajectory is therefore impossible.

Stage B is prospectively amended to a separately identified warm-start
extension. Its checkpoints may be described only as **5K warm-start extension
checkpoints initialized from the Stage A 12K state**. They must not be
described as an exact uninterrupted continuation. This amendment was frozen
before a Stage B smoke, training run, inference run, or result inspection.

## Decision And Rationale

Stage A completed with formal outcome
`PROVISIONAL_2K_ONLY_STAGE_B_REQUIRED`. Only the 2K 12K-kimg checkpoint was
eligible (`14/128`); every 5K and 10K checkpoint was `0/128`. The frozen E009
minimum larger-data role is 5K, so the 2K endpoint cannot unblock E008.

This protocol supersedes only the unexecuted conditional Stage B placeholder
in `configs/e009_stage_a_protocol.json`. Rather than launching a broad 20K
run or extending multiple trajectories, it follows the user-approved
matched-exposure rationale:

```text
2K at 12,000 kimg = 12,000,000 / 2,000 = 6,000 dataset epochs
5K at 30,000 kimg = 30,000,000 / 5,000 = 6,000 dataset epochs
```

The question is whether the existing 5K trajectory enters the frozen
nondegenerate baseline interval by the same approximate example-exposure
count at which the 2K model first became eligible.

## Scope And Non-Goals

Stage B studies additional optimization from the preserved 5K state under a
new, reproducible RNG lineage. It does not:

- train a new 5K model from scratch;
- train or extend the 10K model;
- add a 20K subset;
- alter architecture, optimizer, data, evaluator, criterion,
  eligibility, or pair selection after observing results;
- run donor models, swap windows, E008 conditions, or confirmatory inference.

## Frozen Resume Anchor

The completed Stage A directory is immutable. Stage B must copy the verified
resume state into a new Stage B scratch/output lineage; it must not write new
files into or remove files from the Stage A directory.

| Artifact | Frozen identity |
| --- | --- |
| Stage A root | `/home/xggh8/data/zw-lab/e009_edm5k_12000kimg` |
| Resume state | `training-state-012000.pt` |
| Resume-state bytes | `669419091` |
| Resume-state SHA-256 | `1073e68c9f45123b53811a12a56a565f296a5ab846212d22e5027bbd81d685f5` |
| 12K EMA snapshot SHA-256 | `a77c19588f9a4f877de961102c16901ee07bbd87e0e4ace6164f92f40c406d58` |
| Stage A config SHA-256 | `e4a076c301e3e330872a6088774a5c7d688a18b0639831f5e250d759282868d8` |
| Training-options SHA-256 | `ca39f38ebb94ee78e2a65ac1f2065efe22c5aa915af3581a2c8e469a649a652f` |
| Run-manifest SHA-256 | `d63dbddaaf06422462e8268ffd40a69459947ee35f82ee1887aaab0b9882c81a` |
| Stage A training commit | `d19c470bc4b547e2bad5488b30892be2814c7b12` |

The immutable new lineage is `e009_stage_b_edm5k_30000kimg`. Its parent
training-state and EMA-snapshot hashes are verified independently before any
output directory is created.

## Frozen Training Contract

| Setting | Value |
| --- | --- |
| Dataset | Existing nested, class-balanced CIFAR-10 5K subset |
| Dataset archive SHA-256 | `1e96a4f7a701bd067f71c725bbe83f1dcd65a750b310f206eee878ce2c07355a` |
| Dataset size | `5000` |
| Warm start | `true` |
| Training seed | `1`, newly frozen for all accessible Stage B RNG sources |
| Start | `12,000` kimg |
| Maximum | `30,000` kimg |
| New checkpoint kimg | `13K, 14K, ..., 30K` |
| New checkpoint count | `18` |
| Checkpoint/state cadence | Every `1,000` kimg |
| Architecture | `ddpmpp`, channels `128`, multipliers `[2,2,2]` |
| Conditioning | Unconditional |
| Loss | EDM loss: `sigma_data=0.5`, `p_mean=-1.2`, `p_std=1.2` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Global/per-GPU batch | `64 / 64` |
| EMA half-life | `500` kimg |
| Dropout | `0.13` |
| Precision | Float32 (`use_fp16=false`) |
| Augmentation | `xflip=false`; no added augmentation |

Initialization is split deliberately: the training network and optimizer load
from `training-state-012000.pt`, while EMA loads from the separately frozen
`network-snapshot-012000.pkl`. The 12,000-kimg starting exposure is reconstructed
from the verified parent artifact identity. Stage B uses a stateful sampler and
a zero-worker DataLoader so sampler position can be serialized without an
unrecoverable worker-prefetch queue.

Every new Stage B state serializes the training network, optimizer, EMA,
explicit progress counters, NumPy RNG state, Torch CPU RNG state, all available
Torch CUDA RNG states, sampler state, and DataLoader-generator state. The state
also records `warm_start=true`, seed `1`, parent hashes, and unavailable RNG
sources explicitly rather than implying they were preserved from Stage A.

The continuation runs through 30K before eligibility is evaluated. There is
no adaptive early stopping based on intermediate memorization measurements.

## Resume Smoke Gate

Before full continuation, a Slurm-only smoke must initialize the new lineage
from the two frozen parent artifacts, run from 12K to exactly 13K, and verify:

- exact resume-state and source/config hashes;
- network and optimizer load from the parent training state;
- EMA loads from the separate parent snapshot;
- seed-1 RNG initialization is reproducible;
- progress starts at 12K and reaches exactly 13K;
- finite loss and gradients;
- unconditional architecture identity;
- readable 13K snapshot and extended training state;
- extended state contains EMA, counters, and every accessible RNG state;
- no Stage A path is modified;
- all output is confined to the immutable Stage B lineage;
- no evaluation, full Stage B continuation, E008 swap, or confirmatory seed is
  used.

Any mismatch stops the workflow before full training.

### Smoke Outcome

Slurm job `15722770` completed with exit code `0:0` in `1:13:39`. It reached
exactly 13K, emitted finite loss, and passed the frozen validator. The 13K
snapshot and extended training state remain external in the immutable Stage B
lineage. Their SHA-256 identities are:

```text
network-snapshot-013000.pkl
6d181c0102e93cfe1c43005675e7c76e01fae18afd402337a35ebc8b2128371c

training-state-013000.pt
8bb1aabceee959ce2478a108b27ad6b34313cf8329cba2b048c9446077a7a130
```

The recursive Stage A identity remained
`2ea46ae65a80aaea8485c6c5c4e869cd9e13075e6d509224e3a69e8ebc6cee7b`
before and after the smoke. No full continuation, baseline evaluation, E008
swap, or confirmatory inference was submitted. See the
[smoke result record](../results/experiment_09_stage_b/smoke_validation.json).
The only stderr messages were nonfatal PyTorch sampler-deprecation,
DDP-stride-performance, and process-group shutdown warnings.

The full continuation is configured to restore the validated schema-v2 13K
state rather than restart from Stage A. Its parent hashes are the snapshot and
state listed above, its validated parent implementation is commit `9e5782f`,
and its first permitted new output is 14K. A separate exact execution commit
and state-load-only preflight are required before submission.

### Full Continuation Outcome

Slurm job `15723871` completed with exit code `0:0` in `20:40:20` and
reached exactly 30K. The validator accepted all snapshots from 13K through
30K, 17 finite continuation-loss records, the unchanged 13K parents, and the
unchanged Stage A manifest. The final snapshot SHA-256 is
`66406526e9024c7280b903cdb1731ae35ed252da87aa1b9932cc9a2d43d446e2`;
the final training-state SHA-256 is
`74779b6f443ab7ef720195e66e29b2c3d36fc8edb13c0f555f08c7a4cd15f98a`.
No baseline evaluation or E008 swap was run by the training job.

## Frozen Baseline Evaluation

After training, freeze and hash exactly these evaluation candidates before
inference:

- 18 new 5K snapshots at `13K..30K`;
- the six E008-eligible EDM-1K snapshots at `2K,4K,6K,8K,10K,12K`.

The six EDM-1K checkpoint hashes are:

```text
2K   3ed209dd1b79e56a4e95d81d21400bdbc2c73e95c7a9d76d49aba0c859ff2258
4K   ec341bbc9cf1e60f0bb768e03eb2812ca75a4407c430571813b98618a9a2bc02
6K   89bead0217ff5f41f79a7ccd7607a4d97152b82fee36febce35eafc0745ad807
8K   fc4c9972915e4609e67804aa693569980c8884bc5f754e049d3b72346cbccbae
10K  20a28ef6f66087f8fcec676dca2d86ff81e859e17ae7507adeb9d49a24d31c0e
12K  e5a7debafcd19191d6557f645216bfcb2e7589922396fd08130e76e3f5388b0a
```

All 24 checkpoints use exactly the same baseline evaluation:

```text
seeds:                 20000..20127
records:               (18 + 6) x 128 = 3,072
sampler:               frozen 18-call pure-Euler, zero churn, no swap
nearest neighbors:     deterministic CPU direct-difference float64
criterion:             d1NN < d2NN / 3
eligibility:           13..115 out of 128, inclusive
reserved seeds:        0..255 untouched
excluded prior seeds:  10000..10127 not reused
```

Both endpoints of a selected pair must be eligible under this same-seed Stage
B evaluation. Historical E008 rates are provenance, not pair-selection inputs.

## Frozen Pair Selection

If at least one new 5K checkpoint and at least one reevaluated EDM-1K
checkpoint are eligible:

1. minimize absolute memorization-rate difference;
2. break ties lexicographically by
   `(new_5k_checkpoint_sha256, edm_1k_checkpoint_sha256)`.

Freeze the selected paths, hashes, counts, rates, confidence intervals,
selection calculation, inventory hashes, seeds, code/config commit, and Slurm
provenance before any E008 preparation.

## Stopping And Outcome Rules

```text
Any 5K checkpoint eligible and an EDM-1K same-seed candidate eligible
    -> freeze the deterministic pair
    -> E009 succeeds
    -> permit a separate E008 execution-preparation review

No 5K checkpoint eligible through 30K
    -> stop
    -> do not extend training automatically
    -> record matched 6,000-epoch exposure as insufficient in this run
    -> require a new training-intervention protocol

5K eligible but no EDM-1K candidate remains eligible on the same seeds
    -> stop without a pair
    -> require a separately reviewed matching strategy
```

Stage B does not authorize E008, further 5K training, 10K continuation, 20K
training, or a new model from scratch.

## Expected Artifacts And Acceptance Gates

- immutable Stage B config and provenance manifest;
- resume-smoke validation;
- 18-checkpoint 5K inventory plus six-checkpoint EDM-1K inventory;
- 3,072 explicit success-or-failure records;
- 24 checkpoint summaries with exact confidence intervals;
- deterministic pair-selection or blocker record;
- baseline-rate curves and complete validation record.

Training implementation cannot begin until the committed configuration pins
the new output paths, resume-copy procedure, config hash, expected source
commit, storage budget, and Slurm resources. Full training cannot begin until
the isolated resume smoke passes. Evaluation cannot begin until all 18 new
checkpoints are complete and their hashes are frozen.

## Compute And Storage Budget

The measured 5K Stage A runtime was about 13.8 hours for 12K kimg. Linear
scaling predicts about 20.8 additional GPU-hours for 18K kimg. Freeze one L40S
task with a maximum 30-hour walltime. Budget approximately 18 GB for the
temporary state trajectory and 5 GB for persistent new snapshots plus the
final state. No checkpoint may be pruned before inventory and evaluation.
