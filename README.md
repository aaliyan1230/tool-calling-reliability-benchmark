# Tool-Calling Reliability Benchmark

Fault-injection benchmark for evaluating tool-calling orchestration policies under production-like failures.

This repo bootstraps Project 1 from the roadmap:
- compare orchestration policies on success, latency, retries, and cost
- simulate realistic failure modes (timeouts, 429s, schema failures, contract drift, network faults)
- generate reproducible artifacts suitable for technical writeups

## Policies Implemented

- `naive_retry`
- `exponential_backoff_jitter`
- `schema_first_fallback`
- `timeout_budget_early_abort`

## Faults Simulated

- `timeout`
- `rate_limit`
- `malformed_schema`
- `contract_drift`
- `network_failure`

## Repo Layout

- `src/tcrb/` core benchmark engine and CLI
- `configs/baseline.json` baseline benchmark config
- `workloads/sample_tasks.json` sample tools + tasks
- `tests/` regression tests for policy behavior and metrics
- `reports/` templates for report + short post

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
tcrb --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

If you do not install the package, you can still run directly:

```bash
PYTHONPATH=src python3 -m tcrb --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

## Outputs

Each run writes to `runs/<label>/`:
- `result.json` full per-task and per-attempt records
- `summary.md` compact results table + failure taxonomy

Core metrics in `summary.md`:
- task success rate
- invalid tool-call rate
- mean and p95 latency
- retries per successful task
- estimated cost per successful task

## Reproducibility Notes

- Seeded simulation (`seed` in config)
- JSON config + workload files are source-controlled
- Deterministic policy logic for same seed/config/workload

## Suggested Experiment Plan

1. Baseline run with `configs/baseline.json`
2. Sweep fault probabilities (mild, medium, severe)
3. Sweep timeout and retry budgets
4. Compare rank shifts by metric (cost-first vs success-first)
5. Use taxonomy to identify which policy fails on which mode

## Extend Next

- Add real provider traces into fault models
- Add async/concurrency and queueing effects
- Add confidence intervals via multi-seed runs
- Add plotting notebook for report-ready figures
