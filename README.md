# Tool-Calling Reliability Benchmark

This repository studies a simple but important question: if one tool-calling planner looks better than another, did it actually become more reliable, or did it just get lucky on a narrow slice of tasks?

Tool-Calling Reliability Benchmark (TCRB) answers that question with reproducible runs, fault injection, side-by-side comparisons, transfer checks, and plain-language reports. The goal is not only to produce scores. The goal is to produce evidence that a human reader can understand.

Detailed project criteria live in `docs/core-goals.md`.

## Important Alignment Note

There are two different kinds of studies in this repo:

- non-LLM planner studies: built-in planners like `policy_native` and `heuristic` choose tools
- LLM-backed studies: `hf_local` uses a real model to score candidate tool choices

The fresh local study shown below is a non-LLM planner study. It validates the benchmark harness, reporting, transfer logic, and study-gate behavior. It does not by itself prove how a real LLM behaves on tool selection.

That second question requires an actual `hf_local` run.

## In Plain English

What this repo does:

- runs planners against tool-calling workloads under controlled failures
- compares a baseline planner against a modified planner
- checks whether the modified planner really improved on the target tasks
- checks whether that improvement transfers to other toolsets or falls apart
- writes markdown reports and plots so you do not need to inspect raw JSON by hand

What this repo has achieved already:

- a full local benchmark and reporting loop
- reusable plot generation from existing artifacts
- asset-backed markdown summaries
- a transfer matrix over four toolsets
- a contributor-friendly minimal planner path for extension work

## The Main Finding From This Repo

The clearest fresh result in this repository is a local study comparing the built-in `heuristic` planner against `policy_native`.

In short:

- the heuristic planner improved target-workload success
- it reduced invalid tool calls on that same target workload
- the effect was strong enough to pass the study-gate check
- but it still failed the transfer-matrix portfolio check

That is not a contradiction. It is the point of the benchmark. A planner can improve on one slice and still fail as a generally better policy.

What this means for alignment:

- we now have a strong harness result
- we do not yet have a headline real-LLM result in the README
- the repo contains a real-LLM path for that next step

## What We Ran

The end-to-end study artifacts are here:

- `runs/study-heuristic-vs-native-base-ms/multi_seed_summary.md`
- `runs/study-heuristic-vs-native-comparison-ms/multi_seed_summary.md`
- `runs/study-heuristic-vs-native-delta/delta-ms.md`
- `runs/study-heuristic-vs-native-matrix/matrix_summary.md`
- `runs/study-heuristic-vs-native-study-gate/study_gate.md`
- `runs/study-heuristic-vs-native-analysis/analysis_summary.md`

The transfer study now covers four toolsets:

- `customer_support`
- `ecommerce_ops`
- `fintech_risk`
- `developer_tools`

The fourth toolset is new in this repo and was added to make transfer claims less tied to the original three domains.

## Findings

### High-Level Outcome

| signal | result | plain-language meaning |
|---|---:|---|
| mean target success delta | +0.1944 | the comparison planner solved more target tasks |
| mean target invalid-call delta | -0.1743 | it made fewer bad tool calls |
| study-gate verdict | PASS | the change is real enough to observe, not just flat noise |
| matrix portfolio verdict | FAIL | the comparison planner is not an across-the-board upgrade |
| rows in transfer matrix | 4 | the transfer check now spans four toolsets |
| worst transfer row | customer_support at -0.1111 | the target transfer threshold was still missed |

### What That Means

For a reader who is new to this kind of benchmark:

- `PASS` in the study gate is good: it means the comparison produced observable movement.
- `FAIL` in the transfer matrix is not a broken run: it means the benchmark caught a limitation in the planner change.
- Seeing both together is useful. It means the benchmark can distinguish “better somewhere” from “better overall.”

### Policy-Level Result

On the target workload, the heuristic planner improved the strongest policies by a visible margin:

- `exponential_backoff_jitter`: success `+0.2778`, invalid calls `-0.2063`
- `naive_retry`: success `+0.2778`, invalid calls `-0.2050`
- `timeout_budget_early_abort`: success `+0.2222`, invalid calls `-0.2857`

So the benchmark is not merely saying “different.” It is showing exactly where the planner got better.

## Visual Walkthrough

If you only look at three things in this repo, look at these.

### 1. Overall Comparison View

This figure summarizes the comparison planner on the target workload across the main metrics.

![Multi-seed overview](runs/study-heuristic-vs-native-analysis/multi_seed_overview.png)

### 2. Did The Comparison Planner Improve?

This figure shows the direct change from `policy_native` to `heuristic` for each policy. In this chart:

- bars above zero in success are good
- bars below zero in invalid calls are also good

![Delta policy view](runs/study-heuristic-vs-native-analysis/delta_policy.png)

### 3. Did The Improvement Transfer?

This figure is the transfer matrix. Values near zero mean little change. Negative values mean the comparison planner got worse on that toolset for that metric.

![Transfer matrix view](runs/study-heuristic-vs-native-analysis/transfer_matrix.png)

The important read is simple: the comparison planner improved the target benchmark summary, but it still missed the target-domain transfer threshold and therefore does not qualify as a clean overall win.

## Methodology

The benchmark is intentionally layered so that each claim has a corresponding proof artifact.

### 1. Run Under Controlled Failures

Planners are evaluated under injected failures such as:

- timeouts
- rate limits
- malformed schemas
- contract drift
- network failures

This makes the benchmark about reliability under stress, not just ideal-path behavior.

### 2. Aggregate Across Seeds

We do not trust one run. Multi-seed summaries reduce the chance that a result is just randomness.

### 3. Compare Baseline vs Comparison Directly

We compute `comparison - base` so the size and direction of a change are explicit.

### 4. Test Transfer Across Toolsets

A planner that improves on one workload but collapses elsewhere should not be treated as a generally better planner. The transfer matrix is where that gets caught.

### 5. Gate On Signal Quality

The study gate exists to reject flatline or weak results. It is there to stop overclaiming.

## How To Read The Artifacts

The repo writes a few stable artifact types under `runs/<label>/`.

| artifact | what it tells you |
|---|---|
| `result.json`, `summary.md` | one run, one planner, one workload |
| `multi_seed.json`, `multi_seed_summary.md` | average behavior across several seeds |
| `delta-ms.json`, `delta-ms.md` | exactly how comparison differs from baseline |
| `matrix.json`, `matrix_summary.md` | whether gains transfer across toolsets |
| `study_gate.json`, `study_gate.md` | whether the result is strong enough to count as evidence |
| `analysis_summary.md` | one compact markdown summary with embedded visuals |

If you want a single artifact to open first, start with `runs/study-heuristic-vs-native-analysis/analysis_summary.md`.

## What Changed In This Project

This repo now includes more than the original benchmark loop.

- reusable plot rendering from stored artifact JSON
- markdown reports that can reference nearby visuals
- a fourth enriched toolset: `developer_tools`
- a minimal deterministic planner path for contributors

Those changes make the project easier to inspect as a research artifact, not just as code.

## Run It Yourself

Local setup is small:

```bash
uv sync --extra dev
uv run pytest
```

To reproduce the kind of study shown above:

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3 --planner-config configs/planners/policy_native.json --label base-ms
```

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3 --planner-config configs/planners/heuristic.json --label comparison-ms
```

```bash
uv run tcrb eval-delta --base-run runs/base-ms/multi_seed.json --comparison-run runs/comparison-ms/multi_seed.json --output-json runs/delta/delta-ms.json --output-report runs/delta/delta-ms.md
```

```bash
uv run python scripts/run_transfer_matrix.py --manifest workloads/enriched/manifest.json --config configs/baseline.json --base-planner-config configs/planners/policy_native.json --comparison-planner-config configs/planners/heuristic.json --target-toolset customer_support --label matrix-study
```

```bash
uv run tcrb study-gate --base-run runs/base-ms/multi_seed.json --comparison-run runs/comparison-ms/multi_seed.json --matrix-json runs/matrix-study/matrix.json --require-matrix-signal --output-json runs/study-gate/study_gate.json --output-report runs/study-gate/study_gate.md
```

```bash
uv run tcrb summarize-run --multi-seed-json runs/comparison-ms/multi_seed.json --delta-json runs/delta/delta-ms.json --matrix-json runs/matrix-study/matrix.json --study-gate-json runs/study-gate/study_gate.json --output-dir runs/analysis --output-report runs/analysis/analysis_summary.md
```

If you prefer a notebook walkthrough, `notebooks/entrypoint_outcomes.ipynb` still exists, but the main reporting path is now artifact-first and works locally without depending on notebook execution.

## Running A Real LLM Study

If the goal is to verify whether a real model picks the relevant tool for a user query, use the HF path.

Relevant files:

- `src/tcrb/hf_planner.py`
- `configs/planners/policy_native.json`
- `configs/planners/hf_qwen2_5_3b_base.json`
- `configs/planners/hf_qwen2_5_3b_comparison.json`
- `scripts/run_northstar_hf.py`

Practical note:

- this is the path that likely wants a GPU such as a T4
- the non-LLM planner studies do not need that GPU

Recommended GPU-backed command:

```bash
uv run python scripts/run_northstar_hf.py --base-planner-config configs/planners/policy_native.json --comparison-planner-config configs/planners/hf_qwen2_5_3b_base.json --run-study-gate --run-summarize
```

## Running The Failure-Aware Finetuning Plan

- `prepare-sft-data` normalizes ShareGPT-style tool-calling datasets into JSONL
- `prepare-dpo-data` reshapes preference datasets such as `ToolPreference`
- `mine-failure-pairs` turns eval failures into `(prompt, chosen, rejected)` DPO rows
- `train-sft` and `train-dpo` lazily import HF training dependencies and run QLoRA-style training

Install the optional ML stack first:

```bash
uv sync --extra dev --extra research
```

Prepare the main SFT set from ToolACE:

```bash
uv run tcrb prepare-sft-data --recipe-config configs/research/sft_toolace_qwen25_3b.json
```

Prepare the masked ToolACE+Hermes variant from the plan:

```bash
uv run tcrb prepare-sft-data --recipe-config configs/research/sft_toolace_hermes_mask20_qwen25_3b.json
```

Run SFT from the prepared JSONL:

```bash
uv run tcrb train-sft --recipe-config configs/research/sft_toolace_qwen25_3b.json
```

If you have BFCL-style eval outputs with `expected_output` and `predicted_output`, mine failure pairs like this:

```bash
uv run tcrb mine-failure-pairs --eval-json runs/bfcl_eval/predictions.json --output-jsonl outputs/research/qwen25-3b-failure-pairs/dpo_train.jsonl
```

You can also prepare the `ToolPreference` backup dataset directly:

```bash
uv run tcrb prepare-dpo-data --recipe-config configs/research/dpo_toolpreference_qwen25_3b.json
```

Run DPO using either mined failures or prepared preference JSONL:

```bash
uv run tcrb train-dpo --recipe-config configs/research/dpo_failure_qwen25_3b.json --dataset-jsonl outputs/research/qwen25-3b-failure-pairs/dpo_train.jsonl
```

Notes:

- the prep and mining commands work on CPU and on macOS
- the training commands are intended for a Linux GPU environment such as Kaggle T4 and use `fp16` + optional 4-bit loading
- the recipe files under `configs/research/` are concrete starting points for the exact SFT, masking, and DPO stages in the attached plan

The HF northstar script now refuses to run if base and comparison planner configs are operationally identical, unless you explicitly allow that. That guard exists to prevent meaningless base-vs-comparison LLM studies.

At the moment, the most meaningful default real-LLM comparison in this repo is `policy_native` vs `hf_local`. A true model-vs-model or base-vs-finetuned HF comparison would need a genuinely different HF planner config, not just a different config name.

## If You Want To Extend The Repo

The easiest place to start is the minimal planner path:

- `configs/planners/minimal.json`
- `examples/custom_planner_minimal.py`
- `docs/extending-planners.md`

Current planner families in the repo:

- `policy_native`
- `heuristic`
- `minimal`
- `stochastic`
- `replay`
- `command`
- `hf_local`

`hf_local` is the advanced path. The rest of the benchmark and reporting workflow can be run locally without GPU requirements.
