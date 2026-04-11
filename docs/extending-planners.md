# Extending Planners

This repo is a benchmark first, not a general agent framework. The right way to add a planner is to make it easy to compare fairly against the existing baselines and to keep its behavior deterministic enough that deltas are interpretable.

## What A Planner Is Here

The planner contract lives in `src/tcrb/planner.py` as the `ToolPlanner` protocol. A planner only needs one operational method:

```python
def choose_tool(
    self,
    *,
    task: TaskSpec,
    workload: Workload,
    policy: str,
    attempt_number: int,
    attempted_tools: set[str],
    last_status: str | None,
    rng: random.Random,
) -> str: ...
```

Inputs are intentionally benchmark-oriented:

- `task`: the task being solved
- `workload`: the available tool catalog and tasks
- `policy`: the retry or fallback policy currently under evaluation
- `attempt_number`: the current attempt index
- `attempted_tools`: tools already tried for this task
- `last_status`: the most recent failure status, if any
- `rng`: seeded randomness for reproducible stochastic behavior

The output is only the chosen tool name.

## Current Planner Loading Path

Planner configs are loaded from JSON and dispatched through `load_tool_planner()` and `planner_from_dict()` in `src/tcrb/planner.py`.

Current built-in planner types are:

- `policy_native`
- `heuristic`
- `minimal`
- `stochastic`
- `replay`
- `command`
- `hf_local`

The simplest way to add a new planner today is:

1. Add a new planner class in `src/tcrb/planner.py`.
2. Register it in `planner_from_dict()`.
3. Add a config JSON under `configs/planners/`.
4. Add tests that exercise both behavior and config loading.

This repo does not yet expose a plugin entrypoint for external planners. Keep the first version local and testable.

## Minimal Shape For A New Planner

Use a deterministic or mostly deterministic planner first. That keeps base-vs-comparison claims stable.

```python
@dataclass
class MinimalPlanner:
    planner_id: str = "minimal"

    def choose_tool(
        self,
        *,
        task: TaskSpec,
        workload: Workload,
        policy: str,
        attempt_number: int,
        attempted_tools: set[str],
        last_status: str | None,
        rng: random.Random,
    ) -> str:
        del policy, attempt_number, attempted_tools, last_status, rng
        return task.primary_tool
```

That is intentionally boring. The benchmark value comes from measuring how planner behavior changes outcomes under faults, not from building a large planner abstraction first.

## Config Shape

Planner config JSON is simple. For example:

```json
{
  "type": "heuristic",
  "name": "heuristic",
  "prefer_schema": true,
  "avoid_reusing_attempted": true
}
```

For a new planner, keep config fields explicit and typed. Avoid hidden defaults that make comparisons ambiguous.

## Determinism Rules

If a planner is not deterministic enough, the benchmark will confuse planner variance with real signal.

Good defaults:

- use the provided `rng` instead of global randomness
- make fallback ordering explicit
- keep heuristic scores stable for the same task and toolset
- do not let logging, time, or environment state affect tool choice

For stochastic planners, make the stochastic knobs part of config so they can be reproduced exactly.

## Metrics That Matter In This Repo

When evaluating a new planner, focus on the metrics the benchmark already treats as first-class:

- task success rate
- invalid tool-call rate
- mean latency and p95 latency
- retries per successful task
- cost per successful task

Improving one metric by destroying transfer or invalid-call behavior is not a clean win here.

## Recommended Comparison Flow

Start with a built-in baseline and one new planner config.

Base run:

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3 --planner-config configs/planners/policy_native.json --label planner-base
```

Comparison run:

```bash
uv run tcrb multi-seed --config configs/baseline.json --workload workloads/sample_tasks.json --seeds 1,2,3 --planner-config configs/planners/heuristic.json --label planner-comparison
```

Delta:

```bash
uv run tcrb eval-delta --base-run runs/planner-base/multi_seed.json --comparison-run runs/planner-comparison/multi_seed.json --output-json runs/planner-comparison/delta-ms.json --output-report runs/planner-comparison/delta-ms.md
```

Transfer matrix:

```bash
uv run python scripts/run_transfer_matrix.py --manifest workloads/enriched/manifest.json --config configs/baseline.json --base-planner-config configs/planners/policy_native.json --comparison-planner-config configs/planners/heuristic.json --target-toolset customer_support --max-tasks 0 --label planner-matrix
```

Study gate:

```bash
uv run tcrb study-gate --base-run runs/planner-base/multi_seed.json --comparison-run runs/planner-comparison/multi_seed.json --matrix-json runs/planner-matrix/matrix.json --require-matrix-signal --output-json runs/planner-comparison/study_gate.json --output-report runs/planner-comparison/study_gate.md
```

Artifact-first summary:

```bash
uv run tcrb summarize-run --multi-seed-json runs/planner-comparison/multi_seed.json --delta-json runs/planner-comparison/delta-ms.json --matrix-json runs/planner-matrix/matrix.json --study-gate-json runs/planner-comparison/study_gate.json --output-dir runs/planner-analysis --output-report runs/planner-analysis/analysis_summary.md
```

## Positioning `hf_local`

`hf_local` is an advanced path. Use it after you have a deterministic benchmark comparison working with a simpler planner.

Reasons:

- model loading adds environment complexity
- schema-guard behavior can flatten meaningful adaptation if configured badly
- adapter and base-model compatibility can fail independently of benchmark logic

If you are validating the benchmark loop itself, start with `policy_native`, `heuristic`, or a small new deterministic planner first.

## What To Test When Adding A Planner

At minimum add:

- one unit test for the planner decision rule
- one config-loading test through `planner_from_dict()` or `load_tool_planner()`
- one regression-style comparison using existing CLI flows if the planner changes selection logic materially

Good test locations:

- `tests/test_planner.py`
- `tests/test_cli_analysis.py`
- `tests/test_experiments.py`

## Practical Advice

If a new planner looks good only on target deltas, do not stop there. Run the matrix and study-gate checks. In this repo, a planner change is only interesting if it produces visible signal and survives the transfer sanity check well enough to support the claim you want to make.