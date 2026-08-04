# E009: Intermediate-Dataset Model Design

> **DESIGN PROPOSAL — NO TRAINING OR PILOT EXECUTED**

## Motivation

The E008 baseline-only preflight found six eligible EDM-1K intermediate
checkpoints but no eligible EDM-50K checkpoint. Every EDM-50K snapshot from
`0` through `40,000` kimg produced `0/128` memorized samples. Extending the
same 50K run is therefore not the preferred next control.

E009 will search for a larger-data model with nonzero, nondegenerate baseline
memorization using intermediate CIFAR-10 subset sizes. This is model-pair
design infrastructure, not an E008 swap experiment.

## Proposed Sweep

- Dataset sizes: `2K`, `5K`, `10K`, and `20K` CIFAR-10 training examples.
- Match the existing clean-room EDM architecture, sampler, loss, optimizer,
  preprocessing, unconditional setting, and other training controls.
- Save checkpoints frequently enough to observe entry into and exit from the
  frozen eligible interval rather than evaluating only the final model.
- Freeze ordered subset manifests, hashes, training configs, checkpoint
  cadence, and a new 128-seed pilot set before training or evaluation.
- The new pilot seeds must be disjoint from both `10000..10127` and reserved
  confirmatory seeds `0..255`.
- Preserve the eligibility rule `13..115` memorized samples out of `128` under
  the unchanged criterion `d1NN < d2NN / 3`.

## Model-Pair Gate

A future larger-data checkpoint can be paired only after an independent
baseline-only pilot passes all provenance and determinism checks. Among
eligible candidates, choose a cross-role pair according to a prospectively
frozen rate-matching rule, preferably against one of the six eligible EDM-1K
checkpoints identified by E008 preflight.

No E008 target or control condition may be evaluated during model selection.
Confirmatory seeds remain unavailable until a model pair is frozen.

## Current Status

No subset manifest, training config, seed list, checkpoint, training job,
pilot result, or model pair has been created for E009. Those execution choices
require a separate protocol-freeze review before any compute begins.
