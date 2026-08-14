# TCRB v3.5 figures

Generated from the immutable v3.5 result freezes by `scripts/plot_v35_results.py`.

## Suggested captions

- **Fresh holdout signal.** On 12 fresh stale-target-state traces per model (6 unsafe, 6 safe), exact BLOCK and ALLOW rates separate six monitors. Gemini blocks 6/6 unsafe cases with the narrow policy; DeepSeek Pro blocks 0/6 with the broad policy. Luna and DeepSeek Flash both allow 0/6 safe holdout traces under either policy.
- **Failure-family heatmap.** Direct mistakes remain easier than state-dependent stale-target cases. Across all six monitors, stale-target unsafe blocking ranges from 0% to 75%, while safe allowing ranges from 12% to 100%.
- **Development-to-holdout gap.** The six-model view includes the original Luna and DeepSeek Flash comparison as well as the later four-model expansion. Several monitors drop sharply on the fresh state mutations.

All values are loaded from frozen JSON analyses; no values are hand-entered. `ESCALATE` is not counted as exact `BLOCK` or `ALLOW`.
