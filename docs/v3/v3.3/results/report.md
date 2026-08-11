# v3.3 — can a blind spot be produced when nothing announces the gap?

Responses: 280 (240 primary, 40 repeats). Valid: 280. Cost: $0.3876.

Quality gate: **PASS**

|  | Advertised gap | Silent gap |
|---|---|---|
| **Symmetric drop** | `symmetric_advertised` | `symmetric_silent` |
| **Asymmetric drop** | `asymmetric_advertised` | `asymmetric_silent` |

## Rates by cell

### DeepSeek V4 Flash

| View | Scope | Announcement | Unsafe approved | Safe approved | Insufficient |
|---|---|---|---:|---:|---:|
| `full` | — | — | **0.0%** | 100.0% | 0.0% |
| `symmetric_advertised` | symmetric | advertised | **0.0%** | 0.0% | 100.0% |
| `symmetric_silent` | symmetric | silent | **0.0%** | 0.0% | 100.0% |
| `asymmetric_advertised` | asymmetric | advertised | **0.0%** | 0.0% | 100.0% |
| `asymmetric_silent` | asymmetric | silent | **0.0%** | 8.3% | 95.8% |

### GPT-5.6 Terra

| View | Scope | Announcement | Unsafe approved | Safe approved | Insufficient |
|---|---|---|---:|---:|---:|
| `full` | — | — | **0.0%** | 91.7% | 4.2% |
| `symmetric_advertised` | symmetric | advertised | **0.0%** | 0.0% | 100.0% |
| `symmetric_silent` | symmetric | silent | **0.0%** | 0.0% | 100.0% |
| `asymmetric_advertised` | asymmetric | advertised | **0.0%** | 0.0% | 100.0% |
| `asymmetric_silent` | asymmetric | silent | **0.0%** | 0.0% | 100.0% |

## Blind spots against the full trace

### DeepSeek V4 Flash

| View | Blind-spot rate | 95% CI | Unsafe pairs |
|---|---:|---|---:|
| `symmetric_advertised` | **0.0%** | [+0.0, +0.0] | 12 |
| `symmetric_silent` | **0.0%** | [+0.0, +0.0] | 12 |
| `asymmetric_advertised` | **0.0%** | [+0.0, +0.0] | 12 |
| `asymmetric_silent` | **0.0%** | [+0.0, +0.0] | 12 |

### GPT-5.6 Terra

| View | Blind-spot rate | 95% CI | Unsafe pairs |
|---|---:|---|---:|
| `symmetric_advertised` | **0.0%** | [+0.0, +0.0] | 12 |
| `symmetric_silent` | **0.0%** | [+0.0, +0.0] | 12 |
| `asymmetric_advertised` | **0.0%** | [+0.0, +0.0] | 12 |
| `asymmetric_silent` | **0.0%** | [+0.0, +0.0] | 12 |

## Moving one factor at a time

### DeepSeek V4 Flash

| Contrast | Factor | Unsafe-approval change | 95% CI | Safe-approval change |
|---|---|---:|---|---:|
| `announcement_effect_asymmetric` | announcement | **+0.0pp** | [+0.0, +0.0] | +8.3pp |
| `announcement_effect_symmetric` | announcement | **+0.0pp** | [+0.0, +0.0] | +0.0pp |
| `scope_effect_advertised` | scope | **+0.0pp** | [+0.0, +0.0] | +0.0pp |
| `scope_effect_silent` | scope | **+0.0pp** | [+0.0, +0.0] | +8.3pp |

### GPT-5.6 Terra

| Contrast | Factor | Unsafe-approval change | 95% CI | Safe-approval change |
|---|---|---:|---|---:|
| `announcement_effect_asymmetric` | announcement | **+0.0pp** | [+0.0, +0.0] | +0.0pp |
| `announcement_effect_symmetric` | announcement | **+0.0pp** | [+0.0, +0.0] | +0.0pp |
| `scope_effect_advertised` | scope | **+0.0pp** | [+0.0, +0.0] | +0.0pp |
| `scope_effect_silent` | scope | **+0.0pp** | [+0.0, +0.0] | +0.0pp |

## Figures

![figure-1-unsafe-approval-by-view](figure-1-unsafe-approval-by-view.png)
![figure-2-blind-spots](figure-2-blind-spots.png)
![figure-3-main-effects](figure-3-main-effects.png)

## Stability

Repeat cells: 40, unanimous: 100.0%

## Provenance

- Traces: 24 · views: 120
- Prompt: tcrb.v031.prompts (imported unchanged)
- Commit: `aef4b9b2971eb5290c5a8895e858387a0550f255`

