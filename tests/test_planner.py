import random

from tcrb.models import TaskSpec, ToolSpec, Workload
from tcrb.planner import (
    FinetunedPlanner,
    HeuristicPlanner,
    PolicyNativePlanner,
    ReplayPlanner,
    StochasticPlanner,
    planner_from_dict,
)


def _workload() -> Workload:
    return Workload(
        tools={
            "a": ToolSpec(
                name="a", base_latency_ms=20, jitter_ms=0, schema_fields=["x"]
            ),
            "b": ToolSpec(
                name="b", base_latency_ms=25, jitter_ms=0, schema_fields=["x", "y"]
            ),
        },
        tasks=[
            TaskSpec(
                task_id="t",
                primary_tool="a",
                fallback_tools=["b"],
                required_schema=["x", "y"],
            ),
        ],
    )


def test_policy_native_prefers_schema_tool_for_schema_first_policy():
    workload = _workload()
    task = workload.tasks[0]
    planner = PolicyNativePlanner()

    chosen = planner.choose_tool(
        task=task,
        workload=workload,
        policy="schema_first_fallback",
        attempt_number=1,
        attempted_tools=set(),
        last_status=None,
        rng=random.Random(1),
    )

    assert chosen == "b"


def test_heuristic_prefers_schema_and_untried_tool():
    workload = _workload()
    task = workload.tasks[0]
    planner = HeuristicPlanner()

    chosen = planner.choose_tool(
        task=task,
        workload=workload,
        policy="naive_retry",
        attempt_number=2,
        attempted_tools={"a"},
        last_status="invalid_schema",
        rng=random.Random(2),
    )

    assert chosen == "b"


def test_stochastic_can_emit_off_catalog_tool():
    workload = _workload()
    task = workload.tasks[0]
    planner = StochasticPlanner(
        off_catalog_probability=1.0, hallucinated_tool_name="ghost"
    )

    chosen = planner.choose_tool(
        task=task,
        workload=workload,
        policy="naive_retry",
        attempt_number=1,
        attempted_tools=set(),
        last_status=None,
        rng=random.Random(3),
    )

    assert chosen == "ghost"


def test_replay_uses_sequence_then_fallback():
    workload = _workload()
    task = workload.tasks[0]
    planner = ReplayPlanner(task_tool_sequence={"t": ["b"]})

    chosen_first = planner.choose_tool(
        task=task,
        workload=workload,
        policy="naive_retry",
        attempt_number=1,
        attempted_tools=set(),
        last_status=None,
        rng=random.Random(4),
    )
    chosen_second = planner.choose_tool(
        task=task,
        workload=workload,
        policy="naive_retry",
        attempt_number=2,
        attempted_tools={"b"},
        last_status="network_failure",
        rng=random.Random(4),
    )

    assert chosen_first == "b"
    assert chosen_second == "a"


def test_finetuned_planner_from_dict_resolves_command_template():
    planner = planner_from_dict(
        {
            "type": "finetuned",
            "name": "llama-ft",
            "base_model": "llama3.1:8b",
            "adapter_path": "finetuned-models/llama-ft-v1/adapter",
            "base_command": "python scripts/planners/ollama_tool_selector.py --model {base_model} --lora {adapter_path}",
            "timeout_seconds": 9.0,
        }
    )

    assert isinstance(planner, FinetunedPlanner)
    assert planner.planner_id == "llama-ft"
    assert planner.timeout_seconds == 9.0
    assert planner.resolved_command() == (
        "python scripts/planners/ollama_tool_selector.py "
        "--model llama3.1:8b --lora finetuned-models/llama-ft-v1/adapter"
    )


def test_finetuned_planner_from_dict_parses_strict_mode():
    planner = planner_from_dict(
        {
            "type": "finetuned",
            "name": "llama-ft-strict",
            "base_model": "llama3.1:8b",
            "adapter_path": "finetuned-models/llama-ft-v1/adapter",
            "base_command": "python scripts/planners/ollama_tool_selector.py --model {base_model} --lora {adapter_path}",
            "timeout_seconds": 9.0,
            "strict_mode": True,
        }
    )

    assert isinstance(planner, FinetunedPlanner)
    assert planner.strict_mode is True
