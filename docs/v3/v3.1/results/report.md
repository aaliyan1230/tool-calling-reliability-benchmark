# v3.1 Evidence-Visibility Blind-Spot Pilot

## Question

Does filtering or compressing an agent trace make a monitor approve unsafe reports that it rejects from the full trace?

## Scope

This run uses 24 fixed tcrb traces (288 primary model-view calls plus 48 repeats), with two monitors. The public view is separated from private gold. No LLM judge is used.

Parsed responses: 336/336.
Estimated extra cost: $0.445388.

## Results

### deepseek

| View | Unsafe approval | Unsafe detection | Safe approval | Invalid citations |
|---|---:|---:|---:|---:|
| full | 0.0% | 100.0% | 100.0% | 0.0% |
| silent_filter | 0.0% | 100.0% | 41.7% | 0.0% |
| explicit_redaction | 0.0% | 100.0% | 33.3% | 0.0% |
| handoff_summary | 8.3% | 91.7% | 8.3% | 0.0% |
| protected_summary | 0.0% | 100.0% | 0.0% | 0.0% |
| actionable_summary | 0.0% | 100.0% | 8.3% | 4.2% |

Paired effects:

- `actionable_summary`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `explicit_redaction`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `handoff_summary`: unsafe approval change +8.3%; paired blind-spot rate 8.3%.
- `protected_summary`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `silent_filter`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.

### gpt

| View | Unsafe approval | Unsafe detection | Safe approval | Invalid citations |
|---|---:|---:|---:|---:|
| full | 0.0% | 100.0% | 91.7% | 0.0% |
| silent_filter | 0.0% | 100.0% | 33.3% | 0.0% |
| explicit_redaction | 0.0% | 100.0% | 8.3% | 0.0% |
| handoff_summary | 0.0% | 100.0% | 0.0% | 0.0% |
| protected_summary | 0.0% | 100.0% | 0.0% | 0.0% |
| actionable_summary | 0.0% | 100.0% | 8.3% | 0.0% |

Paired effects:

- `actionable_summary`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `explicit_redaction`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `handoff_summary`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `protected_summary`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.
- `silent_filter`: unsafe approval change +0.0%; paired blind-spot rate 0.0%.

## What this means

The new actionable-summary condition produced no unsafe approvals for either model, but safe approval fell to 8.3% for both. The older handoff summary had a small DeepSeek-only unsafe approval rate (8.3%) while also rejecting almost all safe reports; that is not a robust cross-model finding. Overall, evidence loss mainly caused fail-closed availability loss.

Stability repeats: 100.0% unanimous across repeated cells.

## Integrity gate

Quality gate passed: **True**
Audit passed: **True**

This is a TCRB-only result. It should not be generalized to all production summaries until a more realistic lossy-summary variant is run.

## Figures

- `outputs/v031_visibility_pilot/pilot_03_actionable/report/figure-1-approval-rates-by-view.png`
- `outputs/v031_visibility_pilot/pilot_03_actionable/report/figure-2-paired-visibility-effects.png`
