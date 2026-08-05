# E009 Stage A Evaluation Execution Amendment

Date frozen: 2026-08-05

Status: **AUTHORIZED FOR INVENTORY, SMOKE, AND BASELINE-ONLY PILOT**

This dated amendment operationalizes the already frozen E009 Stage A protocol.
It does not change the dataset subsets, training runs, sampler, seeds,
memorization criterion, eligibility interval, or pair-selection rule.

## Research Question And Hypothesis

The question is whether the completed 2K, 5K, or 10K matched EDM trajectories
contain an intermediate checkpoint with a nonzero, nondegenerate no-swap
memorization baseline. The preregistered hypothesis is that at least one 5K or
10K checkpoint will satisfy the inclusive `13..115 / 128` eligibility gate and
can be paired with an eligible historical EDM-1K checkpoint at a similar pilot
rate. A null result triggers the already specified Stage B review; it does not
authorize Stage B automatically.

## Completed Training Inputs

- Training job: `15673597`, all three tasks completed `0:0`.
- Training source commit: `d19c470bc4b547e2bad5488b30892be2814c7b12`.
- Persistent roots:
  - `/home/xggh8/data/zw-lab/e009_edm2k_12000kimg`
  - `/home/xggh8/data/zw-lab/e009_edm5k_12000kimg`
  - `/home/xggh8/data/zw-lab/e009_edm10k_12000kimg`
- Required snapshots per root: `network-snapshot-000000.pkl` through
  `network-snapshot-012000.pkl` at exactly 1,000-kimg cadence.

## Execution Order And Gates

1. Submit the inventory-only Slurm job.
2. Require exactly 39 persistent, unique, hash-recorded, readable,
   unconditional EMA checkpoints with matching architecture identity.
3. Import and commit the compact inventory, pool manifest, and identity
   sidecar before inference.
4. Run an isolated smoke over the 12K checkpoint from each role and seeds
   `20000,20001` twice. Require exact equality of all per-sample fields.
5. Submit the three-role full pilot only after smoke passes.
6. Require exactly 4,992 explicit checkpoint/seed records and deterministic
   summarization.
7. Apply the frozen eligibility and pair-selection rules without revision.

The manifest cannot contain its own SHA-256 without a self-referential hash.
Therefore `checkpoint_pool_identity.json` is the authoritative sidecar that
records the final manifest SHA-256 and inventory SHA-256.

## Frozen Inventory Result

Inventory job `15720448` completed `0:0` at repository commit
`b7dd105a0620f061b187e3f1f1e850c808450f0b`. All 39 expected checkpoints
were accepted: 13 each for 2K, 5K, and 10K at exact `0..12K` cadence. Every
checkpoint was readable, contained the unconditional EMA network, shared one
architecture identity, and resolved to persistent storage. No checkpoint was
rejected or silently substituted.

- Inventory SHA-256:
  `0cf77c6bb3087ebc45f42eb10861516e43e38a9f72f35e24cd78b3d297ef67ad`
- Manifest SHA-256:
  `c2aa09841509c826a3288ca99a9838b2a07a001885c681a081c736b35bd12548`
- Compact artifacts: [`results/experiment_09_stage_a/`](../results/experiment_09_stage_a/)

The pool is frozen before inference. Smoke and the full pilot remain
unexecuted at this inventory milestone.

## Scientific Invariants

- Pilot seeds are exactly `20000..20127`.
- Seeds `0..255` and `10000..10127` remain untouched.
- Sampling is an 18-call pure-Euler no-swap trajectory with zero churn.
- No donor checkpoint, swap interval, target, or control is accepted.
- Nearest neighbors use deterministic CPU direct-difference `float64`
  arithmetic and stable reference-position tie breaking.
- Memorization remains `d1NN < d2NN / 3`.
- Eligibility remains `13..115` memorized samples out of 128.
- Pair selection minimizes absolute pilot-rate difference, then prefers the
  larger new dataset, then uses the checkpoint-SHA lexical tuple.
- E008 remains unexecuted throughout E009.

## Expected Outputs

Inventory gate:

- `candidate_checkpoint_inventory.csv`: 39 rows.
- `candidate_pool_manifest.json`: frozen pool and execution contract.
- `checkpoint_pool_identity.json`: inventory and manifest hashes.
- Three 13-checkpoint role shards under `shards/`.

Smoke gate:

- Two independently generated six-row pilot CSVs.
- Exact comparison and validation JSON.
- No swap fields and no confirmatory seeds.

Full pilot:

- 4,992 explicit per-sample rows.
- 39 checkpoint summaries with Clopper-Pearson 95% intervals.
- Eligibility curves for 2K, 5K, and 10K.
- Validation, outcome, pair-selection, and run-provenance records.

## Stop Conditions

Stop before inference if inventory count, cadence, path, hash, loadability,
EMA, architecture, conditioning, or manifest validation fails. Stop before the
full pilot if either smoke rerun differs. Do not start Stage B or E008 swaps in
this workflow.

## Storage Deviation

The original protocol's 12 GB wording did not match observed aggregate
storage. Scratch usage was approximately 11 GB per run and persistent usage
approximately 3.4 GB per run. Available storage was sufficient, no checkpoint
was pruned, and no scientific setting changed. The original protocol text is
preserved; this amendment records the observed deviation.

## Operational Inventory Failure

Inventory job `15720430` failed before checkpoint inspection or inference.
The path guard compared the canonical `/mnt/pixstor/data/...` resolution of a
candidate against the unresolved `/home/xggh8/data/...` alias. The correction
canonicalizes both paths and adds a symlink-alias regression test. The final
inventory output was not created, and no scientific setting changed.
