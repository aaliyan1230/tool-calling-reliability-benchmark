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
uv sync --extra dev
uv run pytest
uv run tcrb --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

If you want to run module entrypoints explicitly:

```bash
uv run python -m tcrb --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

## CLI Commands

Single run (default behavior):

```bash
uv run tcrb run --config configs/baseline.json --workload workloads/sample_tasks.json --label baseline
```

Multi-seed aggregate with confidence intervals:

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3,4,5 --label ms-baseline
```

Scenario sweep (each scenario is run as multi-seed):

```bash
uv run tcrb sweep --base-config configs/baseline.json --sweep-config configs/sweeps/fault_levels.json --workload workloads/sample_tasks.json --label sweep-fault-levels
```

Finetune dataset export from an existing run payload:

```bash
uv run tcrb finetune-data --input-json runs/baseline/result.json --output-dir finetuned-models/training --validation-split 0.2 --seed 42
```

Base-vs-finetuned delta comparison (supports both result.json and multi_seed.json):

```bash
uv run tcrb eval-delta --base-run runs/ms-model-base-target/multi_seed.json --finetuned-run runs/ms-model-ft-target/multi_seed.json --output-json runs/ms-model-ft-target/delta.json
```

## Planner Abstraction (Provider-Agnostic)

The benchmark now supports a model-planner abstraction so you can evaluate orchestration policies against different planner behaviors without binding to any closed provider SDK.

Planner config path is passed with `--planner-config`.

Included planner types:
- `policy_native`: baseline planner that mirrors existing policy logic
- `heuristic`: schema-aware and latency-aware deterministic chooser
- `stochastic`: probabilistic chooser with tunable off-catalog tool-call rate
- `replay`: fixed per-task tool sequence playback
- `command`: shell command adapter (bridge to any external/open model runner)
- `finetuned`: command planner template with `{base_model}` and `{adapter_path}` substitution

Example runs:

```bash
uv run tcrb run --planner-config configs/planners/heuristic.json --label run-heuristic
uv run tcrb multi-seed --planner-config configs/planners/stochastic_lowhalluc.json --seeds 1,2,3,4,5 --label ms-stochastic
```

### Command Planner Contract

`command` planner receives JSON on stdin with task context and must return either:
- plain tool name on stdout, or
- JSON with `{"tool_name": "..."}`

This lets you plug in local runners (for example, open-weight models served via local infra) while keeping the benchmark core unchanged.

## Outputs

Each run writes to `runs/<label>/`:
- `result.json` full per-task and per-attempt records
- `summary.md` compact results table + failure taxonomy

Per-attempt records now include `planner_id` for model/planner attribution.

Multi-seed runs write:
- `multi_seed.json` per-seed policy metrics + aggregate stats
- `multi_seed_summary.md` mean +/- CI95 metric table

Sweep runs write:
- `sweep.json` all scenarios with nested multi-seed results
- `sweep_summary.md` scenario comparison table

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
6. Compare open local model planners on the same workload/policies
7. Fine-tune each open model on a target toolchain trace set
8. Re-evaluate base vs fine-tuned models on both target and open toolchains
9. Quantify in-domain gains vs out-of-domain regressions

Concrete commands:

```bash
uv run tcrb multi-seed --seeds 1,2,3,4,5 --label ms-baseline
uv run tcrb sweep --sweep-config configs/sweeps/fault_levels.json --label sweep-fault-levels
uv run tcrb sweep --sweep-config configs/sweeps/budget_tradeoff.json --label sweep-budget
```

### Fine-Tuning Extension Plan (Base vs Adapted)

Use this extension when you want to test whether model adaptation to a specific toolchain improves tool-calling reliability without overfitting.

1. Prepare two workloads:
   - target toolchain workload used for adaptation/evaluation (`target`)
   - held-out open toolchain workload used for transfer check (`open`)
2. For each base model, run baseline evaluation on both workloads.
3. Build supervised fine-tuning data from target toolchain traces:
   - input: planner payload (task, attempted tools, last status, available tools)
   - output: strict `{"tool_name":"..."}` label
4. Fine-tune with LoRA/QLoRA per model family (same train budget for fairness).
5. Register fine-tuned checkpoints behind planner configs (same command planner contract).
6. Re-run the benchmark matrix for base and fine-tuned variants on both workloads.
7. Report `delta` metrics by variant:
   - target delta: `finetuned - base` on success/latency/cost
   - transfer delta: `finetuned - base` on open workload
8. Flag regressions where target gain is positive but open transfer degrades materially.

Recommended run label pattern:
- base target: `ms-<model>-base-target`
- base open: `ms-<model>-base-open`
- ft target: `ms-<model>-ft-target`
- ft open: `ms-<model>-ft-open`

## Plot Frontier

Generate a publication-ready scatter showing success vs p95 latency, with point size mapped to cost per success.

```bash
uv run python scripts/plot_frontier.py --input runs/ms-baseline/multi_seed.json --output runs/ms-baseline/frontier.png
```

## Notebook Analysis

The repo includes a notebook for local open-model comparison:

- `analysis/ollama_model_comparison.ipynb`

It uses the existing `tcrb` helpers (`load_workload`, `load_benchmark_config`, `load_tool_planner`, `run_multi_seed`) and can either:
- analyze cached output in `analysis/ollama_open_models_baseline_s10.json`, or
- run fresh local simulations when `RUN_BENCHMARKS = True`.

The notebook writes a combined frontier figure to:

- `analysis/ollama_open_models_baseline_s10_frontier.png`

## Extend Next

- Add real provider traces into fault models
- Add async/concurrency and queueing effects
- Add confidence intervals via multi-seed runs
- Add plotting notebook for report-ready figures
- Add base-vs-fine-tuned model benchmarking across target and open toolchains

## Enriched Multi-Toolset Workflow

To align with a realistic north star, this repo now supports richer mock tool ecosystems with question-level expected tool-call labels.

Generate enriched mock toolsets and eval cases:

```bash
uv run python scripts/generate_enriched_toolsets.py
```

This writes:

- `workloads/enriched/*.json` (3 toolsets, each 16 tools, 18 tasks)
- `workloads/eval_cases/*_eval_cases.json` (question + expected tool labels)
- `workloads/enriched/manifest.json` (index of generated toolsets)

Run benchmark on one toolset:

```bash
uv run tcrb run --config configs/baseline.json --workload workloads/enriched/customer_support.json --label cs-baseline
```

Score actual calls against expected tool labels:

```bash
uv run tcrb eval-cases-score --result-json runs/cs-baseline/result.json --eval-cases-json workloads/eval_cases/customer_support_eval_cases.json --output-json runs/cs-baseline/eval_case_score.json
```

Recommended experiment loop per toolset:

1. Run base model planner on one enriched toolset.
2. Export finetune data from the same toolset run.
3. Fine-tune one model (LoRA/QLoRA) for that single toolset.
4. Re-run benchmark on same toolset and compare deltas.
5. Run open transfer checks on the other toolsets.
6. Report both policy metrics and eval-case first-tool accuracy.

### Cross-Toolset Transfer Matrix (Base vs Finetuned)

To evaluate one target-toolset finetune against other toolsets (open transfer):

```bash
uv run python scripts/run_transfer_matrix.py \
   --manifest workloads/enriched/manifest.json \
   --config configs/baseline.json \
   --base-planner-config configs/planners/ollama_qwen2_5_3b.json \
   --ft-planner-config configs/planners/finetuned_qwen2_5_3b_example.json \
   --target-toolset customer_support \
   --label matrix-customer-support
```

For strict smoke runs with heavier local model planners, limit scope first:

```bash
uv run python scripts/run_transfer_matrix.py \
   --manifest workloads/enriched/manifest.json \
   --config configs/baseline.json \
   --base-planner-config configs/planners/ollama_qwen3_4b_strict.json \
   --ft-planner-config configs/planners/ollama_qwen3_4b_strict.json \
   --target-toolset customer_support \
   --toolsets customer_support \
   --max-tasks 1 \
   --label matrix-qwen3-4b-strict-smoke
```

Reference planner configs for this flow:

- `configs/planners/ollama_qwen2_5_3b.json`
- `configs/planners/finetuned_qwen2_5_3b_example.json`

Outputs are written under `runs/<label>/`:

- per-toolset base/ft benchmark artifacts
- per-toolset eval-case scores
- `matrix.json` with gate-ready deltas
- `matrix_summary.md` with a compact transfer table

Eval-case gate defaults:

- target split: delta first-tool >= +0.03 and delta sequence >= +0.03
- open splits: delta first-tool >= -0.03 and delta sequence >= -0.03

### Create Real `qwen3:4b-ft` Tag From Kaggle QLoRA Adapter

After Kaggle QLoRA PEFT training, download adapter artifacts (for example `outputs/ft-notebook/final`) to local and create an Ollama finetuned tag:

```bash
uv run python scripts/create_ollama_adapter_tag.py \
   --base-model qwen3:4b \
   --adapter-dir outputs/ft-notebook/final \
   --tag qwen3:4b-ft \
   --modelfile models/ollama/Modelfile.qwen3-4b-ft \
   --create
```

Then run strict base-vs-ft matrix:

```bash
uv run python scripts/run_transfer_matrix.py \
   --manifest workloads/enriched/manifest.json \
   --config configs/baseline.json \
   --base-planner-config configs/planners/ollama_qwen3_4b_strict.json \
   --ft-planner-config configs/planners/finetuned_qwen3_4b_strict_tag.json \
   --target-toolset customer_support \
   --label matrix-qwen3-4b-fttag-strict
```
