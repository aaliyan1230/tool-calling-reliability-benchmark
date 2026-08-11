# v3.2 — is fail-closed about missing values or about summary packaging?

Responses: 224 (192 primary, 32 repeats). Valid: 224. Cost: $0.3053.

Quality gate: **PASS** · Integrity audit: **PASS**

## The design

|  | Values kept | Values stripped |
|---|---|---|
| **Raw packaging** | `full` | `raw_projection` |
| **Summary packaging** | `lossless_summary` | `actionable_summary` |

`full` and `actionable_summary` are carried over unchanged from v3.1, so their
numbers double as a replication check.

## Rates by cell

### DeepSeek V4 Flash

| View | Packaging | Values | Safe approved | Unsafe approved | Insufficient evidence |
|---|---|---|---:|---:|---:|
| `full` | raw | kept | 100.0% | 0.0% | 0.0% |
| `lossless_summary` | summary | kept | 100.0% | 0.0% | 0.0% |
| `raw_projection` | raw | stripped | 8.3% | 0.0% | 91.7% |
| `actionable_summary` | summary | stripped | 8.3% | 0.0% | 91.7% |

### GPT-5.6 Terra

| View | Packaging | Values | Safe approved | Unsafe approved | Insufficient evidence |
|---|---|---|---:|---:|---:|
| `full` | raw | kept | 100.0% | 0.0% | 0.0% |
| `lossless_summary` | summary | kept | 91.7% | 0.0% | 4.2% |
| `raw_projection` | raw | stripped | 8.3% | 0.0% | 91.7% |
| `actionable_summary` | summary | stripped | 8.3% | 0.0% | 91.7% |

## Moving one factor at a time

### DeepSeek V4 Flash

| Contrast | Factor moved | Safe-approval change | 95% CI | Unsafe-approval change | Blind spots |
|---|---|---:|---|---:|---:|
| `packaging_effect_values_present` | packaging | +0.0pp | [+0.0, +0.0] | +0.0pp | 0.0% |
| `packaging_effect_values_stripped` | packaging | +0.0pp | [+0.0, +0.0] | +0.0pp | 0.0% |
| `content_effect_raw` | content | -91.7pp | [-100.0, -83.3] | +0.0pp | 0.0% |
| `content_effect_summary` | content | -91.7pp | [-100.0, -83.3] | +0.0pp | 0.0% |

### GPT-5.6 Terra

| Contrast | Factor moved | Safe-approval change | 95% CI | Unsafe-approval change | Blind spots |
|---|---|---:|---|---:|---:|
| `packaging_effect_values_present` | packaging | -8.3pp | [-16.7, +0.0] | +0.0pp | 0.0% |
| `packaging_effect_values_stripped` | packaging | +0.0pp | [+0.0, +0.0] | +0.0pp | 0.0% |
| `content_effect_raw` | content | -91.7pp | [-100.0, -83.3] | +0.0pp | 0.0% |
| `content_effect_summary` | content | -83.3pp | [-100.0, -71.4] | +0.0pp | 0.0% |

## Figures

![figure-1-two-by-two-approval](figure-1-two-by-two-approval.png)
![figure-2-one-factor-contrasts](figure-2-one-factor-contrasts.png)
![figure-3-main-effects](figure-3-main-effects.png)

## Stability

Repeat cells: 32, unanimous rate: 100.0%

## Provenance

- Traces: 24 · views: 96
- Prompt: tcrb.v031.prompts (imported unchanged)
- Providers: tcrb.v031.providers (imported unchanged)
- Commit: `fbdee986bed1dffcd6cfdbe78d4cb82693fc1b2a`

