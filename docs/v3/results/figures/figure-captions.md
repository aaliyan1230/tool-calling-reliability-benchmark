# Figure captions

## Figure 1 — Label/evidence interaction

Within each pair, the payload and independent evidence are unchanged; only the visible self-label changes.

- **DeepSeek V4 Flash (16 base cases):** Baseline clean cases: 6.2% flagged with `verified` versus 25.0% with `warning`; Evidence-first clean cases: 0.0% flagged with `verified` versus 6.2% with `warning`.
- **GPT-5.6 Terra (16 base cases):** Baseline clean cases: 6.2% flagged with `verified` versus 12.5% with `warning`; Evidence-first clean cases: 0.0% flagged with `verified` versus 0.0% with `warning`.

Corrupted cases provide the safety check: both prompts should keep flagging them.

## Figure 2 — Paired prompt improvement

Points are oriented so positive values are improvements. Whiskers are 95% bootstrap intervals from resampling the 16 base cases while keeping each prompt pair together. Both false-alarm reductions point in the desired direction, but their intervals touch zero; this pilot is too small for a firm population claim.

## Figure 3 — Repeatability

Four pre-selected base cases—one per domain—were each evaluated three times per matched condition. The chart reports decision unanimity and majority-vote performance. This is a small stability check, not a confidence interval or a new independently sampled test set.

## Shared caveat

These are deterministic simulated traces, not production logs. The core pilot contains 16 base cases, so small percentage changes can represent only one or two cases.
