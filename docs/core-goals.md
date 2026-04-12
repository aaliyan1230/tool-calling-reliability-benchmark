# Core Goals

This repository has two goals and one definition of done: outputs must be reproducible and interpretable.

Important scope note:
- planner-policy studies and LLM-backed studies are not the same thing
- planner-policy studies validate the benchmark harness and decision logic under controlled failures
- LLM-backed studies validate whether a real model selects the relevant tool for the user query it sees
- the repository should support both modes clearly without presenting one as evidence for the other

## Goal 1: Reliability Benchmarking Under Failures

Measure tool-calling orchestration reliability under controlled fault injection.

Achievement criteria:
- Policies are comparable under timeout, rate-limit, malformed-schema, contract-drift, and network-failure conditions.
- Runs produce stable artifacts from checked-in configs.
- Tradeoffs across success rate, invalid calls, latency, retries, and cost are explicit.
- For LLM-backed runs, expected-tool eval cases verify whether the model picked the relevant tool for the use case.

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
- Reports explicitly state whether evidence came from non-LLM planners, LLM-backed planners, or both.

Primary artifacts:
- `delta-ms.json`, `delta-ms.md`
- `matrix.json`, `matrix_summary.md`
- `study_gate.json`, `study_gate.md`

## Practical Definition Of Done

A run is considered complete when:
- commands execute from repository configs without manual patching,
- generated markdown explains outcomes clearly,
- JSON and markdown outputs agree on verdicts and key deltas.
- and the run description makes it explicit whether tool choice came from a real LLM or a non-LLM planner.
