# Experiment 4 Cutoff Decision

## Status

**Finalized as a single-reviewer qualitative visual decision.**

One human reviewer inspected all 20 frozen CIFAR-10 examples at every candidate
cutoff in \(\{2,3,4,5,6,8\}\). The reviewer judged the operational
structure/detail separation to be visually acceptable around \(r=4,5,6\).

The originally planned two-independent-reviewer scoring procedure was not
completed. Reviewer A and Reviewer B CSV templates remain blank and unchanged.
No ordinal scores, qualification counts, inter-rater agreement, or blinded
review result are claimed.

## Final Decision

The operational reference cutoff is

\[
r_\star = 4.
\]

Experiment 5 must use \(r=3\) and \(r=5\) as its primary lower and upper
sensitivity cutoffs. It may additionally report \(r=6\) as an extended
sensitivity check, but \(r=6\) does not replace either primary sensitivity
cutoff.

```text
status: finalized_single_reviewer_visual_decision
reference_cutoff: 4
primary_sensitivity_cutoffs: [3, 5]
optional_extended_sensitivity_cutoff: 6
review_method: single_reviewer_qualitative_visual_inspection
two_reviewer_scoring_completed: false
```

## Rationale

The reviewer selected \(r=4\) because it is the smallest cutoff in the visually
acceptable range. Across the frozen examples:

- low-pass reconstructions at \(r=4\) generally retain recognizable object
  identity and coarse spatial layout;
- the exact complementary high-pass components are primarily concentrated on
  edges and localized detail;
- \(r=5\) and \(r=6\) provide nearby visually acceptable comparisons rather
  than evidence for one universal semantic boundary.

This rationale concerns an operational frequency split only. It does not make
low frequency semantically identical to general structure or high frequency
semantically identical to fine detail.

## Protocol Disclosure

The frozen protocol specified two independent reviewers, per-row ordinal
scores, per-reviewer qualification thresholds, and an adjacent-cutoff
stability rule. Those scoring and inter-rater gates were not executed.
Therefore:

- this decision must not be described as passing the frozen two-reviewer
  qualification rule;
- it must not be described as an inter-rater, blinded, or independently
  replicated review result;
- no reviewer CSV values may be inferred from this qualitative judgment;
- the difficult and unattractive examples remain part of the canonical E004
  outputs.

The human decision supersedes the pending status only. It does not alter the
frozen dataset, image set, FFT convention, masks, candidate cutoffs, display
normalization, or numerical reconstruction checks.

## Scope

This record freezes the operational cutoffs before denoiser curves are
examined. It does not implement Experiment 5 and contains no Experiment 5
results. Any later E005 interpretation must report \(r=3,4,5\) together and
may include \(r=6\) only as the declared extended sensitivity check.
