# Tool-Calling Reliability Benchmark

Tool-Calling Reliability Benchmark (TCRB) is a benchmark-and-repair workflow for tool-routing reliability.

It is built to answer a stricter question than "did the score go up?":

If we repair a planner so it chooses the right tool more often on the target workload, does that improvement survive under controlled failures, across multiple seeds, and across other toolsets?

Detailed project criteria live in `docs/core-goals.md`.

Artifact bundle: https://doi.org/10.5281/zenodo.20071002

## Project Trajectory

This repository records one continuous reliability study. The work began with first-tool routing repair and expanded into full action traces, executable tools, controlled failures, recovery prompts, and recovery training.

### Phase 1: Routing Repair

The initial study used Qwen2.5-3B-Instruct and a small DPO dataset of 157 first-tool preference examples. On the target customer-support workload, DPO raised success from `83.33%` to `90.74%`, raised first-tool accuracy from `83.33%` to `100%`, and lowered invalid tool calls from `13.67%` to `7.19%`.

That improvement was local: ecommerce first-tool accuracy fell from `100%` to `88.89%`, while fintech and developer tools were unchanged. This established the central question for the next phase: does a repair survive outside the workload it was trained on?

### Phase 2: Failure-Recovery Extension

The benchmark was extended from choosing only a tool name to generating complete actions: tool name, arguments, final answers, clarifications, and safe aborts. It now runs 160 tasks across customer support, ecommerce, fintech, and developer tools, with five controlled tool-failure types and explicit seen/unseen domain and hazard splits.

The current detailed report is [`reports/reliability-study/README.md`](reports/reliability-study/README.md). The compact result artifact is [`reports/reliability-study/results.json`](reports/reliability-study/results.json).

### Current Findings

| condition | clean | faulted |
|---|---:|---:|
| normal Qwen baseline | `31.25%` | `10.16%` |
| recovery instructions | `29.38%` | `12.03%` |
| corrected SFT adapter | `30.63%` | `10.00%` |

The recovery instructions are the strongest completed intervention. They improve failure handling while costing a small amount of clean-task accuracy. Corrected supervised fine-tuning (SFT: training on examples of the desired recovery action) removes an earlier formatting failure but does not beat the simpler prompt. Preference training (DPO: teaching the model to prefer a good action over a bad action) and Gemini reviewer experiments remain future work, not completed findings.

### Current Position

The evidence supports a reliability benchmark and a cautious empirical finding, not a claim that training has solved tool recovery. The highest-value next step is multi-seed replication of the baseline and recovery-prompt results. After that, DPO or a Gemini reviewer should be evaluated as separate interventions.

## v3–v3.5 Monitor-Safety Experiments

The later TCRB work studies a different question: can a monitor catch an unsafe tool write when it must connect the policy, the full trace, and the current database state?

The research progressed in small pilots. Early versions tested self-labels and missing evidence. They found that monitors usually failed closed: they rejected safe work instead of approving unsafe work. v3.3 also showed that the first hand-built cases could not express the intended blind spot cleanly.

v3.4 rebuilt the data pipeline around official τ-bench v1.0.1 traces. It added blinded human labels, constrained LLM mutations, replay checks, downstream consistency checks, and frozen private gold. v3.5 used 26 matched airline and retail pairs across four monitor models.

The clearest current result is that direct mistakes are easy, while stale-target mistakes are hard. On a fresh six-pair holdout, broad-policy unsafe blocking was `2/6` for GPT-5.6 Terra, `0/6` for DeepSeek V4 Pro, `3/6` for Qwen3.7 Plus, and `5/6` for Gemini 3.6 Flash. A simple deterministic target-join check solved all 28 stale-target traces.

Read the full report: [`docs/v3/TCRB_V3_V3.5_REPORT.md`](docs/v3/TCRB_V3_V3.5_REPORT.md).

## Earlier Routing Study

The strongest result in this repo is a targeted router-repair run on a real HF planner:

- base model: `Qwen/Qwen2.5-3B-Instruct`
- repair step: DPO on a small first-tool preference dataset
- training set size: `157` preference rows
- target domain: `customer_support`

### What improved

| signal | base | repaired | delta |
|---|---:|---:|---:|
| target success across seeds `11,13,17` | `0.8333` | `0.9074` | `+0.0741` |
| invalid tool-call rate | `0.1367` | `0.0719` | `-0.0649` |
| target first-tool accuracy (`customer_support`) | `0.8333` | `1.0000` | `+0.1667` |

### What did not transfer cleanly

| toolset | base first-tool acc | repaired first-tool acc | delta |
|---|---:|---:|---:|
| `customer_support` | `0.8333` | `1.0000` | `+0.1667` |
| `ecommerce_ops` | `1.0000` | `0.8889` | `-0.1111` |
| `fintech_risk` | unchanged | unchanged | `0.0000` |
| `developer_tools` | unchanged | unchanged | `0.0000` |

## Why This Matters

- the repair fixed the specific failure mode it targeted
- the gain replicated across multiple seeds, not one lucky run
- invalid calls dropped while success rose—the operational direction that matters
- the benchmark surfaced the transfer limit rather than hiding it, which is what a reliability benchmark is supposed to do

What this repo shows:

1. a small targeted repair can measurably improve first-tool routing on a real workload
2. that gain holds up under stricter methodology than a single benchmark score
3. transfer must be measured explicitly—target improvement and cross-domain robustness are separate properties

## What TCRB Actually Does

TCRB combines four pieces:

- controlled workload execution with injected failures
- baseline-vs-comparison evaluation
- multi-seed aggregation
- transfer checks across multiple toolsets

The repo supports both:

- benchmark-only studies with built-in planners such as `policy_native` and `heuristic`
- HF-backed studies with `hf_local`, including adapter-backed comparisons and targeted repair runs

## Methodology

The methodology is intentionally conservative.

### 1. Targeted task signal

We measure whether the comparison planner improves the first tool it selects and whether task success also improves.

### 2. Error quality

We track invalid tool calls directly, so a planner does not get credit for reaching higher success through noisier behavior.

### 3. Multi-seed evaluation

We aggregate across seeds to reduce the chance that one favorable run becomes the entire story.

### 4. Transfer checks

We test the same change on multiple toolsets:

- `customer_support`
- `ecommerce_ops`
- `fintech_risk`
- `developer_tools`

That is what turns a local win into a meaningful reliability claim, or shows where the claim stops.

## Repo Outputs

The stable artifacts under `runs/<label>/` are:

| artifact | purpose |
|---|---|
| `summary.md`, `result.json` | one planner on one workload |
| `multi_seed_summary.md`, `multi_seed.json` | aggregated behavior across seeds |
| `delta-ms.md`, `delta-ms.json` | direct comparison between base and repaired planner |
| `matrix_summary.md`, `matrix.json` | transfer behavior across toolsets |
| `analysis_summary.md` | compact artifact-first summary with nearby visuals |

If you want one place to start, open:

- `runs/study-heuristic-vs-native-analysis/analysis_summary.md` for the benchmark harness story
- the Zenodo artifact bundle above for the latest end-to-end repair bundle

## Benchmark Foundation

Before the HF router-repair result, this repo already established the harness on built-in planners.

That earlier work matters because it validated the reporting loop, transfer matrix, and study-gate logic before using the same framework on a real HF planner.

Two representative visuals from that benchmark foundation:

![Multi-seed overview](assets/readme/multi_seed_overview.png)

![Transfer matrix view](assets/readme/transfer_matrix.png)

Those figures are from the benchmark-only path, not from the later HF repair run, but they show the same reliability framing the repo now applies to LLM-backed planner studies.

## What Changed In The Repo

The repo now includes the full execution path for the HF repair workflow:

- strict HF planner configs with explicit candidate-scope controls
- adapter-backed planner comparisons
- direct first-tool probe and matrix scripts
- DPO dataset construction for router repair
- Kaggle packaging and execution helpers for GPU-backed runs
- training-side fixes for adapter continuation and DPO argument compatibility

Key files:

- `src/tcrb/hf_planner.py`
- `src/tcrb/research.py`
- `scripts/run_northstar_hf.py`
- `scripts/run_hf_choice_probe.py`
- `scripts/run_hf_choice_matrix.py`
- `scripts/build_router_dpo_dataset.py`
- `scripts/kaggle_tcrb.py`
- `kaggle/runner.py`

## Minimal Workflows

Most of the old command wall is not needed day to day. These are the only workflows worth surfacing.

### 1. Local sanity and tests

```bash
uv sync --extra dev
uv run pytest
```

### 2. Strict HF comparison

Use this when you want a clean base-vs-adapter first-tool study with explicit candidate controls:

```bash
uv run python scripts/run_northstar_hf.py --base-planner-config configs/planners/hf_qwen2_5_3b_base_taskscope_strict.json --comparison-planner-config configs/planners/hf_qwen2_5_3b_comparison_taskscope_strict.json --skip-matrix --run-study-gate --run-summarize
```

### 3. Kaggle-backed repair run

Use this when you want the GPU execution path:

```bash
uv run python scripts/kaggle_tcrb.py dataset-version
uv run python scripts/kaggle_tcrb.py push
uv run python scripts/kaggle_tcrb.py watch
uv run python scripts/kaggle_tcrb.py output --output-dir outputs/kaggle
```

## Extending The Repo

If you want to extend planners or add new workloads, start with:

- `configs/planners/minimal.json`
- `examples/custom_planner_minimal.py`
- `docs/extending-planners.md`

Current planner families:

- `policy_native`
- `heuristic`
- `minimal`
- `stochastic`
- `replay`
- `command`
- `hf_local`
