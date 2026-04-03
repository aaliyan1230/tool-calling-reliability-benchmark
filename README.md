# Tool-Calling Reliability Benchmark

Fault-injection benchmark for tool-calling orchestration reliability.

This repository focuses on two outcomes:

- reliable benchmarking under realistic failures
- reproducible evaluation with interpretable signals

Core workflow is CLI-first; notebook-driven paths are intentionally removed.

## Core Goals

1. Measure reliability under controlled fault injection.
2. Prove evaluation signal quality with reproducible artifacts.

See detailed criteria in `docs/core-goals.md`.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
```

## Core Workflow

Single benchmark run:

```bash
uv run tcrb run --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

Multi-seed benchmark:

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3,4,5 --label ms-baseline
```

Pairwise comparison deltas:

```bash
uv run tcrb eval-delta --base-run runs/ms-base/multi_seed.json --comparison-run runs/ms-comparison/multi_seed.json --output-json runs/ms-comparison/delta.json --output-report runs/ms-comparison/delta.md
```

Study-gate signal checks:

```bash
uv run tcrb study-gate --base-run runs/ms-base/multi_seed.json --comparison-run runs/ms-comparison/multi_seed.json --matrix-json runs/matrix-hf/matrix.json --require-matrix-signal --output-json runs/ms-comparison/study_gate.json --output-report runs/ms-comparison/study_gate.md --fail-on-violation
```

Transfer matrix:

```bash
uv run python scripts/run_transfer_matrix.py --manifest workloads/enriched/manifest.json --config configs/baseline.json --base-planner-config configs/planners/hf_qwen2_5_3b_base.json --comparison-planner-config configs/planners/hf_qwen2_5_3b_comparison.json --target-toolset customer_support --toolsets customer_support,ecommerce_ops,fintech_risk --max-tasks 18 --label matrix-hf
```

End-to-end north-star run:

```bash
uv run python scripts/run_northstar_hf.py
```

## Interpretable Outputs

Core outputs are written under `runs/<label>/` and are human-readable:

- `result.json` or `multi_seed.json`: raw benchmark metrics
- `summary.md` or `multi_seed_summary.md`: readable benchmark summaries
- `delta-ms.json` and `delta-ms.md`: base-vs-comparison deltas
- `matrix.json` and `matrix_summary.md`: transfer and generalization checks
- `study_gate.json` and `study_gate.md`: pass/fail signal-quality checks

## Planner Types

- `policy_native`
- `heuristic`
- `stochastic`
- `replay`
- `command`
- `hf_local`
