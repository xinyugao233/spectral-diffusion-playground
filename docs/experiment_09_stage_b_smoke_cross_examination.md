# E009 Stage B Resume Smoke: Cross-Examination

## 1. Exact Run Identity

- Experiment: E009 Stage B 5K warm-start smoke.
- Repository: `spectral-diffusion-playground`, commit `9e5782f09a3e024c298dc5ce8da1c0f44c9b4fbd`.
- Entrypoint: `experiments/09_stage_b_warm_start.py`.
- Launcher: `scripts/e009_stage_b_resume_smoke.slurm`.
- Config: `configs/e009_stage_b_edm5k_13000kimg_smoke.yaml`.
- Job: `15722770`, `COMPLETED`, exit `0:0`, elapsed `1:13:39`.
- Output: `/home/xggh8/data/zw-lab/e009_stage_b_edm5k_30000kimg`.
- Artifacts: 13K EMA snapshot and extended training state.

## 2. Training Bookkeeping

- One L40S; global/per-GPU batch `64/64`; Adam at `0.001`.
- 15,625 optimizer steps advanced exposure from 12,000 to 13,000 kimg.
- Increment: 1,000 kimg = 1 mimg = 200 epochs over the 5K subset.
- Parent optimizer state was restored; parent EMA was loaded separately.
- Stage B RNG seed was newly initialized at `1`; this is not uninterrupted
  Stage A randomness.

**Missing training detail that Zhengchao would ask for:** no Stage A RNG state
exists, so exact uninterrupted continuation cannot be reconstructed.

## 3. Data Bookkeeping

- Dataset: frozen nested CIFAR-10 5K archive, 32x32 RGB, unconditional.
- Exactly 5,000 class-balanced training images; no held-out set was evaluated.
- Archive SHA-256: `1e96a4f7a701bd067f71c725bbe83f1dcd65a750b310f206eee878ce2c07355a`.
- The smoke tests execution continuity only; it makes no train/test or
  memorization comparison.

## 4. Exact Critical Definition / Implementation Object

#### Object: Stage B extended warm-start state

- Role in experiment: preserve an exact continuation point within the new
  Stage B lineage.
- Exact definition: network, optimizer, EMA, explicit progress counters,
  NumPy RNG, Torch CPU RNG, all CUDA RNG states, sampler state, and
  DataLoader-generator state.
- Inputs: frozen Stage A training state, frozen Stage A EMA snapshot, seed `1`.
- Output: `training-state-013000.pt`, schema version `2`.
- Thresholds/preprocessing: exact parent hashes; exposure `12K -> 13K`;
  zero-worker DataLoader to avoid an inaccessible prefetch queue.
- Code: `src/spectral_diffusion_playground/e009_warm_start.py`.
- Match to intent: yes, for future continuation inside Stage B; no claim is
  made that it recreates Stage A RNG.
- Likely failures: incorrect parent component, counter mismatch, incomplete
  RNG capture, sampler cursor error, or writes into Stage A.

## 5. Exact Evaluation Protocol

- Runtime validation loaded the 13K snapshot and state on CPU.
- It checked finite tensors/loss, unconditional EMA, schema fields, explicit
  counters, parent identities, and recursive Stage A before/after identity.
- No samples, nearest-neighbor scores, baseline rates, or swaps were computed.
- Evidence: `results/experiment_09_stage_b/smoke_validation.json`.

## 6. Raw Results

- Final exposure: exactly 13,000 kimg.
- Last batch loss mean: `0.07918821275234222`; all recorded loss was finite.
- Snapshot SHA-256: `6d181c0102e93cfe1c43005675e7c76e01fae18afd402337a35ebc8b2128371c`.
- State SHA-256: `8bb1aabceee959ce2478a108b27ad6b34313cf8329cba2b048c9446077a7a130`.
- Stage A before/after manifest SHA-256:
  `2ea46ae65a80aaea8485c6c5c4e869cd9e13075e6d509224e3a69e8ebc6cee7b`.

## 7. What Conclusion The User Wants To Draw

The warm-start implementation is safe and reproducible enough to permit a
separately authorized 13K-to-30K Stage B continuation.

## 8. Does The Evidence Actually Support That Conclusion?

**Yes, for execution readiness only.** All frozen smoke gates passed. The run
does not support any claim about checkpoint eligibility, memorization, or E008.
The unresolved scientific confounder is that Stage B uses a new RNG lineage.

## 9. Zhengchao's Likely Questions

1. Are the parent training-state and EMA hashes exact?
2. Did network and optimizer load from the training state rather than EMA?
3. Did EMA load from the separate snapshot?
4. Why is this not an exact Stage A continuation?
5. Is seed `1` frozen before execution?
6. Is the sampler sequence reproducible and serializable?
7. Does the zero-worker loader alter scientific settings or only operations?
8. Did progress begin at 12K and terminate exactly at 13K?
9. Were any nonfinite losses or gradients observed?
10. Is the 13K state readable and complete?
11. Did any Stage A file change?
12. Were any evaluation or swap seeds consumed?
13. Is the full continuation a separate authorization?
14. What evidence will establish 5K eligibility after 30K?

## 10. Top Failure Modes

1. Treating the warm start as an uninterrupted Stage A trajectory.
2. Restoring network weights but silently resetting optimizer or EMA.
3. Losing sampler position when continuing from 13K.
4. Overwriting Stage A or the validated 13K checkpoint.
5. Interpreting finite training as evidence of memorization eligibility.

## 11. Minimal Disambiguating Experiments

1. Resume the extended 13K state for a tiny isolated state-roundtrip smoke;
   fixed inputs should reproduce the next sampler/RNG initialization exactly.
2. Run the preregistered 13K-to-30K continuation with all settings fixed; this
   tests whether additional warm-start training yields eligible checkpoints.
3. Evaluate all frozen Stage B checkpoints on seeds `20000..20127`; eligibility
   should follow only the preregistered `13..115/128` rule.

## 12. Decision

- **Can we trust the current conclusion?** yes
- **Most important missing detail:** the original Stage A RNG states are unrecoverable.
- **Most important unresolved confounder:** Stage B is a new seed-1 stochastic lineage.
- **Most important definition to verify:** extended state restoration at the next resume.
- **Most important code path to inspect:** `warm_start_training_loop()` state restore/save path.
- **Best next experiment:** separately authorize the frozen 13K-to-30K continuation.
- **What would convince Zhengchao:** exact Stage B state restoration plus the complete preregistered checkpoint inventory and evaluation.
