# Tool-Calling Reliability Benchmark

Fault-injection benchmark for evaluating tool-calling orchestration reliability with reproducible artifacts.

## What This Repo Does

- Benchmarks orchestration policies under realistic failures
- Supports Hugging Face local planner evaluation (base vs adapter)
- Produces multi-seed metrics, deltas, and transfer-matrix outputs

## Policies

- `naive_retry`
- `exponential_backoff_jitter`
- `schema_first_fallback`
- `timeout_budget_early_abort`

## Fault Modes

- `timeout`
- `rate_limit`
- `malformed_schema`
- `contract_drift`
- `network_failure`

## Repo Layout

- `src/tcrb/` benchmark engine and CLI
- `configs/` benchmark and planner configs
- `workloads/` task/tool definitions
- `scripts/run_northstar_hf.py` full HF north-star pipeline runner
- `analysis/northstar_hf_kaggle_runner.ipynb` clean Kaggle notebook path

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run tcrb run --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

## HF North-Star (One Command)

```bash
uv run python scripts/run_northstar_hf.py
```

With post-run anti-flatline gating:

```bash
uv run python scripts/run_northstar_hf.py --run-study-gate --study-gate-require-matrix-signal --study-gate-fail-on-violation
```

Default planner configs:

- base: `configs/planners/hf_qwen2_5_3b_base.json`
- finetuned: `configs/planners/hf_qwen2_5_3b_ft.json`

Outputs are written under `runs/northstar-hf-*`.

## Core CLI Commands

Single run:

```bash
uv run tcrb run --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

Multi-seed aggregate:

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3,4,5 --label ms-baseline
```

Fault sweep:

```bash
uv run tcrb sweep --base-config configs/baseline.json --sweep-config configs/sweeps/fault_levels.json --workload workloads/sample_tasks.json --label sweep-fault-levels
```

Finetune dataset export:

```bash
uv run tcrb finetune-data --input-json runs/baseline/result.json --output-dir finetuned-models/training --validation-split 0.2 --seed 42
```

Base vs finetuned delta:

```bash
uv run tcrb eval-delta --base-run runs/ms-model-base-target/multi_seed.json --finetuned-run runs/ms-model-ft-target/multi_seed.json --output-json runs/ms-model-ft-target/delta.json
```

Anti-flatline study gate:

```bash
uv run tcrb study-gate --base-run runs/ms-model-base-target/multi_seed.json --finetuned-run runs/ms-model-ft-target/multi_seed.json --matrix-json runs/matrix-hf/matrix.json --require-matrix-signal --output-json runs/ms-model-ft-target/study_gate.json --fail-on-violation
```

## Planner Types

- `policy_native`
- `heuristic`
- `stochastic`
- `replay`
- `command`
- `finetuned`
- `hf_local`

## Enriched Toolset Workflow

Generate enriched toolsets and eval cases:

```bash
uv run python scripts/generate_enriched_toolsets.py
```

Run transfer matrix:

```bash
uv run python scripts/run_transfer_matrix.py --manifest workloads/enriched/manifest.json --config configs/baseline.json --base-planner-config configs/planners/hf_qwen2_5_3b_base.json --ft-planner-config configs/planners/hf_qwen2_5_3b_ft.json --target-toolset customer_support --toolsets customer_support,ecommerce_ops,fintech_risk --max-tasks 18 --label matrix-hf
```
