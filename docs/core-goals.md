# Core Goals

This repository has two goals and one definition of done: outputs must be reproducible and interpretable.

## Goal 1: Reliability Benchmarking Under Failures

Measure tool-calling orchestration reliability under controlled fault injection.

Achievement criteria:
- Policies are comparable under timeout, rate-limit, malformed-schema, contract-drift, and network-failure conditions.
- Runs produce stable artifacts from checked-in configs.
- Tradeoffs across success rate, invalid calls, latency, retries, and cost are explicit.

Primary artifacts:
- `result.json`, `summary.md`
- `multi_seed.json`, `multi_seed_summary.md`
- `sweep.json`, `sweep_summary.md`

## Goal 2: Reproducible Evaluation and Signal Validation

Ensure measured differences are real signal, not noise.

Achievement criteria:
- Pairwise base-vs-comparison deltas are reproducible.
- Matrix checks quantify target-vs-open transfer behavior.
- Study-gate checks catch flat or regressive outcomes.

Primary artifacts:
- `delta-ms.json`, `delta-ms.md`
- `matrix.json`, `matrix_summary.md`
- `study_gate.json`, `study_gate.md`

## Practical Definition Of Done

A run is considered complete when:
- commands execute from repository configs without manual patching,
- generated markdown explains outcomes clearly,
- JSON and markdown outputs agree on verdicts and key deltas.
