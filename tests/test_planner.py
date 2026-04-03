import random

import pytest

from tcrb.models import TaskSpec, ToolSpec, Workload
from tcrb.planner import (
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


def test_deprecated_planner_type_is_not_supported():
    with pytest.raises(ValueError):
        planner_from_dict({"type": "deprecated", "name": "deprecated"})
