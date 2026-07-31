# Documentation Index

Protocols freeze decisions before execution. Result documents report the
validated run without changing those decisions. Provenance documents connect
external data and checkpoints to exact hashes and commits.

## E004: Operational CIFAR-10 Cutoff

1. [Frozen cutoff-selection protocol](experiment_04_frequency_cutoff_protocol.md)
2. [Reviewer instructions](experiment_04_reviewer_instructions.md)
3. [Final single-reviewer decision](experiment_04_frequency_cutoff_decision.md)

Result: operational reference `r = 4`, primary sensitivity `r = 3, 5`, and
optional extended sensitivity `r = 6`.

## E005: Spectral Residual Curves

1. [Frozen residual-energy protocol](experiment_05_spectral_residual_protocol.md)
2. [Clean-room model and checkpoint provenance](experiment_05_clean_room_models.md)
3. [Validated residual-curve results](experiment_05_spectral_residual_results.md)

Result: an ordered low-frequency then high-frequency residual transition under
the frozen clean-room setup.

## E006: Transition-Window Swaps

1. [Frozen whole-denoiser swap protocol](experiment_06_transition_swap_protocol.md)
2. [Validated swap results](experiment_06_transition_window_swap_results.md)

Formal outcome: `INCONCLUSIVE` because the EDM-50K no-swap baseline was
degenerate at `0/256`. Descriptively, the low-transition window passed the
frozen influence test in both directions; the high-transition window did not.

## Reading Order

New readers should start with the [root README](../README.md), then read the
E004 decision, E005 results, and E006 results. Reviewers auditing a specific
experiment should read its protocol before its result document and then inspect
the linked compact [`results/`](../results/) and [`figures/`](../figures/)
directories.
