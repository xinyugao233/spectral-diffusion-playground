# Experiment 6: Transition-Window Whole-Denoiser Swap Protocol

## Status

**Frozen protocol only. No Experiment 6 execution has been performed.**

This experiment is a paper-derived clean-room extension of the denoiser
swapping study in *Two Calm Ends and the Wild Middle: A Geometric Picture of
Memorization in Diffusion Models*. The original executed Table 1/Figure 10
swap implementation, checkpoint identities, CIFAR-10 1K ordering, and sampling
seeds were unavailable. This protocol does not claim code identity or exact
numerical reproduction.

The purpose of E006 is to test whether the E005 transition windows are
especially important for trajectory-level memorization under matched
clean-room models. E005 windows are descriptive fixed-sigma residual-energy
transition windows. They are not memorization danger zones before this
intervention study.

## Scientific Question

E005 identified two transition windows on the EDM-1K held-out test residual
curves at the operational reference cutoff \(r=4\):

- low-frequency residual energy -- general-structure proxy: indices `5..11`,
  \(\sigma=12.9101\) down to \(0.585348\);
- high-frequency residual energy -- fine-detail proxy: indices `11..14`,
  \(\sigma=0.585348\) down to \(0.0599473\).

E006 asks:

> Do whole-denoiser swaps over these transition windows change final
> trajectory-level memorization more than width-matched control windows?

The experiment tests a swap intervention. It does not splice Fourier
components of denoiser outputs and does not infer when a model learned a
feature.

## Frozen Model Pair

Use the matched clean-room models validated for E005:

```text
EDM-1K:
/home/xggh8/data/zw-lab/exp_004_standard_edm_n1000_40000kimg_20260415/network-snapshot-040000.pkl
SHA-256: 8e53dd93177c0144d38508c5634ae9ffbce303b6c8209af65085d376ce9026a1

EDM-50K:
/home/xggh8/data/zw-lab/e005_edm50k_matched_40000kimg/network-snapshot-040000.pkl
SHA-256: a355ea67605dea3e2e663e94eb23416ffeb7679757088a68dc6228c03da5a92b
```

Both models are unconditional EDM models trained for `40,000 kimg` with
matched architecture, optimizer, loss, seed, dropout, EMA, precision, batch
size, and preprocessing. They differ only in experiment name and training
subset size (`1000` versus `50000`). They are not the paper's unrecovered
EDM-1K and EDM-50K checkpoints.

All model calls use `class_labels=None`.

## Sampler Convention

The primary sampler is a pure Euler sampler following the paper's stated
trajectory convention. It uses no stochastic churn, no Heun correction, and no
second denoiser call per step.

The frozen 18-point EDM polynomial schedule uses
\(\rho=7\), \(\sigma_{\max}=80\), and \(\sigma_{\min}=0.002\):

```text
index  sigma
0      80
1      57.58598472124816
2      40.78557379650796
3      28.374584604156844
4      19.35245298032523
5      12.91008238075732
6      8.400935309099816
7      5.315194521796382
8      3.256821519765537
9      1.9233398370400518
10     1.088170636545279
11     0.5853481231945422
12     0.29644228447915727
13     0.13951646873101678
14     0.05994731123547159
15     0.022934518372333384
16     0.0075280199627840785
17     0.002000000000000003
```

Sampling appends a terminal \(\sigma_{18}=0\) for the final Euler update.
There are exactly 18 denoiser calls, indexed `0..17`. A swap window refers to
these denoiser-call indices, inclusive.

For base model \(A\), donor model \(B\), and selected window \(W\):

```text
x = sigma[0] * z
for i in 0..17:
    model_i = B if i in W else A
    denoised = model_i(x, sigma[i], class_labels=None)
    derivative = (x - denoised) / sigma[i]
    x = x + (sigma[i + 1] - sigma[i]) * derivative
```

The same model selection applies to every denoiser call in a selected step.
There is no frequency splitting, no partial output mixing, no output clipping,
and no uint8 quantization before memorization evaluation.

Heun sampling may be added only as a separately labeled sensitivity experiment
after the primary Euler result exists. It must not be mixed into the primary
condition table.

## Initial Latents and Seeds

Use exactly 256 initial latent seeds:

```text
sample_seed = 0, 1, 2, ..., 255
```

Each `sample_seed` deterministically produces one Gaussian latent tensor with
shape `(3, 32, 32)`. The initial sampler state is
`x = sigma[0] * z`.

Latent generation must be independent of batch size, condition ordering,
worker rank, and resume behavior. The implementation must generate each
sample's latent from its own recorded seed, not from a single global stream
whose state depends on batching.

The same 256 latent tensors are reused for every no-swap and swap condition.

## Swap Conditions

Whole-denoiser swaps run in both directions:

- base `EDM-1K`, donor `EDM-50K`;
- base `EDM-50K`, donor `EDM-1K`.

No-swap baselines:

| Condition | Base | Donor | Window |
| --- | --- | --- | --- |
| `edm_1k_no_swap` | EDM-1K | none | none |
| `edm_50k_no_swap` | EDM-50K | none | none |

Primary transition windows:

| Window name | Indices | Sigma range | Source |
| --- | --- | --- | --- |
| `low_transition` | `5..11` | `12.9101` to `0.585348` | E005 low-frequency residual |
| `high_transition` | `11..14` | `0.585348` to `0.0599473` | E005 high-frequency residual |
| `combined_transition` | `5..14` | `12.9101` to `0.0599473` | union of low/high windows |
| `paper_medium_reference` | `6..13` | `8.40094` to `0.139516` | Table 1/Figure 10 clean-room reference |

Width-matched controls:

| Target | Width | Pre-window control | Post-window control |
| --- | ---: | --- | --- |
| `low_transition` | 7 | `0..6` | `11..17` |
| `high_transition` | 4 | `7..10` | `14..17` |

Shared endpoints are allowed because the sampler grid is discrete. For
example, `low_transition` and its post-window control both include index `11`;
`high_transition` and its post-window control both include index `14`. This is
frozen before execution and must not be changed after seeing results.

No arbitrary or visually adjusted windows may be introduced after E006
results are inspected.

## Paper Boundary Discrepancy

The paper reports multiple medium-region conventions:

- Table 1 and Figure 10 support a medium window approximately
  \(\sigma \in [0.14, 8.4]\), corresponding to zero-based indices `6..13`.
- Appendix E.6 describes large/medium/small regions as
  `sigma > 5.3`, `sigma in [0.06, 5.3]`, and `sigma < 0.06`, corresponding
  approximately to indices `7..14`.
- Figure 10 text also mentions medium steps `6..13` and a narrower danger-zone
  example `7..10`.

Because the original executed swap configuration was not recovered, E006 uses
`paper_medium_reference = 6..13` as a clean-room Table 1/Figure 10 reference.
This is a compatibility reference, not evidence of original code identity.

## Memorization Evaluation

The reference set is the frozen E005 clean-room 1K CIFAR-10 training subset:

```text
data/e005_cifar10_subset_1k_indices.txt
newline-text SHA-256: 33bb509c48144464a48d3b945cc44c14f880a1e6c6470c283dc0ed65e22b1f29
little-endian int64 SHA-256: f97076ea6db59a96dc81a59d1b573bc8aaecdb8efa1e93c0d79928bfbf8a43f8
```

Generated samples and reference images are represented as unquantized RGB
tensors in `[-1, 1]`, flattened in channel-major `(C, H, W)` order to 3,072
dimensions. Distances are Euclidean L2 distances in this pixel space.

For generated sample \(g\) and reference set \(D_{1K}\), compute the nearest
and second-nearest distances:

\[
d_{1NN}(g, D_{1K}), \qquad d_{2NN}(g, D_{1K}).
\]

The memorization indicator is:

\[
\mathrm{memorized}(g) =
\mathbf{1}\{d_{1NN}(g,D_{1K}) < d_{2NN}(g,D_{1K}) / 3\}.
\]

Each per-sample row must record the first- and second-nearest reference
indices, both distances, the ratio \(d_{1NN}/d_{2NN}\), and the binary
memorization label.

Do not substitute Inception, LPIPS, SSCD, CLIP, perceptual features,
train-versus-test nearest-distance diagnostics, clipping, or uint8 image
space for the primary memorization evaluator.

## Primary Outcomes and Uncertainty

For each condition, report:

- memorized count out of 256;
- memorization rate;
- exact two-sided Clopper-Pearson 95% binomial confidence interval;
- mean, median, and distribution summary of \(d_{1NN}/d_{2NN}\);
- all nearest-neighbor records and generated-sample hashes.

For each swap condition, compare against the no-swap baseline with the same
base model and same 256 seeds. Report paired seed-level changes:

```text
delta_seed = memorized_swap_seed - memorized_baseline_seed
```

Allowed values are `-1`, `0`, and `1`. Report the mean paired change,
discordant-pair counts, a paired bootstrap 95% confidence interval over seeds,
and an exact sign-test p-value over nonzero discordant pairs. These paired
statistics are descriptive safeguards; the practical effect-size rule below is
the frozen decision threshold.

## Effect-Size Rule

A transition-window swap is considered practically influential only if its
absolute memorization-rate change relative to the corresponding no-swap
baseline is at least 10 percentage points and at least 10 percentage points
larger than both of its width-matched controls in the same swap direction.

Formally, for transition window \(T\) with controls \(C_{\mathrm{pre}}\) and
\(C_{\mathrm{post}}\), base model \(A\), and donor model \(B\):

```text
effect(T) = rate(A base, B donor, T) - rate(A no-swap)
```

The practical threshold requires:

```text
abs(effect(T)) >= 0.10
abs(effect(T)) >= abs(effect(C_pre)) + 0.10
abs(effect(T)) >= abs(effect(C_post)) + 0.10
```

The confidence intervals and paired seed-level evidence must support the
reported effect direction. If intervals are too wide to distinguish the
transition from controls, the outcome is not `YES` even if point estimates pass
the threshold.

`combined_transition` and `paper_medium_reference` are reported as secondary
context. They do not replace the width-matched control test for the low and
high transition windows.

## Outcome Classification

Every E006 report must assign exactly one outcome:

| Outcome | Definition |
| --- | --- |
| `YES` | Transition-window swaps produce substantially larger memorization changes than both width-matched controls in both swap directions. |
| `PARTIAL` | A transition effect is dominant in only one swap direction, or only one of the low/high transition windows is influential. |
| `MIXED` | Effects differ by direction, window, or control in a way that does not support a simple conclusion. |
| `NO` | Transition-window effects are not larger than matched controls under the frozen threshold. |
| `INCONCLUSIVE` | Confidence intervals are too wide, no-swap baselines are degenerate, sampler/model failures occur, nearest-neighbor evaluation fails, or convention sensitivity remains unresolved. |

The phrase **candidate memorization danger zone** may be used only after E006
if the outcome is `YES` or `PARTIAL`, and only for the supported window(s).
Even then, the claim is limited to this whole-denoiser swap intervention under
the clean-room setup.

## Required Machine-Readable Outputs

This protocol freezes output contracts only; it does not create them.

### Run Manifest

`results/experiment_06_manifest.json` must contain:

```text
experiment_id, run_id, git_commit, protocol_commit, reproduction_claim,
paper_title, paper_sha256, corrected_plan_sha256,
model_conditions, checkpoint_paths, checkpoint_sha256,
model_source_repositories, model_source_commits, sampler,
sigma_schedule, terminal_sigma, swap_step_semantics, condition_table,
latent_seed_policy, sample_seeds, dataset_archive_identity,
reference_subset_manifest, nearest_neighbor_metric, tensor_domain,
output_clamping, output_quantization, batch_size, device,
dependency_versions, execution_host
```

### Per-Sample Results

`results/experiment_06_per_sample.csv` must contain:

```text
experiment_id,run_id,condition,base_model,base_checkpoint_sha256,
donor_model,donor_checkpoint_sha256,window_name,window_start_index,
window_end_index,window_start_sigma,window_end_sigma,sampler,
sample_seed,batch_index,generated_sample_hash,d1nn,d2nn,
d1nn_reference_index,d2nn_reference_index,d1nn_over_d2nn,
memorized,status,error
```

Stable row order is condition order, then ascending `sample_seed`. The unique
key is `(condition, sample_seed)`.

### Nearest-Neighbor Records

`results/experiment_06_nearest_neighbors.csv` must contain one row per
generated sample and nearest-neighbor rank:

```text
experiment_id,run_id,condition,sample_seed,rank,reference_index,
reference_subset_position,distance
```

Allowed `rank` values are `1` and `2`.

### Condition Summary

`results/experiment_06_condition_summary.csv` must contain:

```text
experiment_id,run_id,condition,base_model,donor_model,window_name,
window_start_index,window_end_index,window_start_sigma,window_end_sigma,
n_samples,memorized_count,memorization_rate,ci95_low,ci95_high,
mean_d1nn_over_d2nn,median_d1nn_over_d2nn,status
```

### Paired Comparisons

`results/experiment_06_paired_comparisons.csv` must contain:

```text
experiment_id,run_id,comparison,base_model,donor_model,swap_condition,
baseline_condition,window_name,control_type,n_pairs,
baseline_rate,swap_rate,rate_difference,paired_mean_delta,
paired_ci95_low,paired_ci95_high,discordant_negative,
discordant_positive,discordant_zero,sign_test_p_value,
passes_practical_threshold,status
```

### Confidence Intervals and Outcome

`results/experiment_06_outcome.json` must contain:

```text
experiment_id,run_id,outcome,decision_rule_version,
effect_size_threshold_pp,condition_results,paired_results,
transition_vs_control_results,unsupported_danger_zone_language_check,
failure_summary,interpretation
```

### Failure Report

`results/experiment_06_failures.csv` must contain every non-`ok` generated
sample, nearest-neighbor evaluation failure, nonfinite sample, hash mismatch,
or missing row. No failed row may be silently dropped from summaries.

## Required Figures

Figures are generated only after all machine-readable outputs validate:

1. memorization rates for all no-swap and swap conditions;
2. paired memorization-rate changes from each corresponding no-swap baseline;
3. low/high transition windows versus their width-matched controls;
4. \(d_{1NN}/d_{2NN}\) ratio distributions by condition;
5. representative generated sample and nearest-neighbor pairs;
6. concise comparison with the `paper_medium_reference` condition.

Qualitative samples are selected deterministically after results exist:

1. for each primary transition condition, list seeds in ascending order;
2. choose the first two newly memorized samples relative to that condition's
   no-swap baseline;
3. choose the first two no-longer-memorized samples, if present;
4. choose the first two unchanged memorized samples, if present;
5. choose the first two unchanged non-memorized samples as fillers;
6. if a category has fewer than two samples, leave it short and report the
   missing category rather than hand-picking replacements.

This rule prevents selecting only attractive or unusually clear examples.

## Validation Gates Before Implementation

Implementation cannot begin until this protocol specifies:

- sampler convention and terminal-sigma handling;
- denoiser-step index semantics;
- all windows and controls;
- model identities and checkpoint hashes;
- seed policy and sample count;
- nearest-neighbor tensor representation;
- confidence intervals and paired comparisons;
- practical effect-size threshold;
- outcome classification;
- output schemas;
- qualitative-selection rule;
- failure-reporting policy.

All gates are specified in this document. Passing this documentation phase does
not authorize E006 implementation, sampling, Slurm submission, or E007 work.

## Unresolved Ambiguities

- The original executed Table 1/Figure 10 swap source was not recovered.
- The paper has inconsistent medium-boundary descriptions between Table
  1/Figure 10 and Appendix E.6.
- The paper states Euler sampling, while the official EDM sampler commonly
  includes a second-order correction. E006 freezes pure Euler as the primary
  clean-room convention.
- The clean-room checkpoints are matched and validated but are not the paper's
  original checkpoints.
- The clean-room 1K subset is frozen and documented but is not the unrecovered
  paper subset.
- Pixel-space \(d_{1NN}<d_{2NN}/3\) follows the paper's stated criterion but
  may not capture all perceptual memorization cases.
- If no-swap baseline rates are too close to 0 or 1 for informative paired
  changes, the outcome must be `INCONCLUSIVE` rather than rescued with
  post-hoc windows or metrics.
