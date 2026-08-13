# TCRB v3.5 figures

Generated from the immutable v3.5 result freezes by `scripts/plot_v35_results.py`.

## Suggested captions

- **Fresh holdout signal.** On 12 fresh stale-target-state traces per model (6 unsafe, 6 safe), exact BLOCK and ALLOW rates separate the monitors. Gemini blocks 6/6 unsafe cases with the narrow policy; DeepSeek blocks 0/6 with the broad policy. This is a model-dependent generalization gap, not a uniform failure.
- **Failure-family heatmap.** Direct mistakes are blocked at 100% in this aggregate. State-dependent stale-target cases are harder: unsafe blocking ranges from 0% to 75%, while safe allowing ranges from 62% to 100%.
- **Development-to-holdout gap.** The same monitor can look strong on familiar development cases and fall on fresh target-state mutations. Under the broad policy, DeepSeek drops from 6/8 to 0/6 unsafe cases blocked; Gemini drops from 7/8 to 5/6.

All values are loaded from frozen JSON analyses; no values are hand-entered. `ESCALATE` is not counted as exact `BLOCK` or `ALLOW`.
