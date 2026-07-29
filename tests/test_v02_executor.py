from tcrb.v02.agent import ReplayAgent
from tcrb.v02.executor import EpisodeConfig, run_episode
from tcrb.v02.types import Clarify, FaultSchedule, FinalAnswer, TaskDef, ToolCall, ToolDef


def test_clarification_terminates_missing_information_episode():
    task = TaskDef(
        task_id="TEST-001",
        domain="test",
        user_query="Check the account balance.",
        category="missing_information",
        available_tools=[],
    )

    trace = run_episode(
        agent=ReplayAgent(action_sequence=[Clarify("Which account ID?")]),
        task=task,
        tool_defs={},
    )

    assert len(trace.steps) == 1
    assert isinstance(trace.steps[0].parsed_action, Clarify)
    assert trace.final_response == "Which account ID?"
    assert trace.diagnostic_labels == []


def test_fault_waits_for_its_target_tool_call():
    def execute(arguments, state, rng):
        return {"ok": True}

    tools = {
        name: ToolDef(
            name=name,
            description=name,
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object"},
            executor=execute,
        )
        for name in ("other", "target")
    }
    task = TaskDef(
        task_id="TEST-002",
        domain="test",
        user_query="Use both tools.",
        category="tool_required",
        available_tools=["other", "target"],
    )
    schedule = FaultSchedule(
        task_id=task.task_id,
        fault_type="execution_error",
        step_index=0,
        tool_name="target",
        config={},
    )

    trace = run_episode(
        agent=ReplayAgent(
            action_sequence=[
                ToolCall("other", {}),
                ToolCall("target", {}),
                FinalAnswer("finished"),
            ]
        ),
        task=task,
        tool_defs=tools,
        fault_schedules=[schedule],
    )

    assert trace.steps[0].observation.status == "success"
    assert trace.steps[1].observation.status == "execution_error"
    assert trace.faults_applied == ["execution_error"]


def test_agent_can_recover_from_invalid_arguments_within_step_budget():
    def execute(arguments, state, rng):
        return {"customer_id": arguments["customer_id"]}

    tool = ToolDef(
        name="lookup",
        description="lookup",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
        output_schema={"type": "object"},
        executor=execute,
    )
    task = TaskDef(
        task_id="TEST-003",
        domain="test",
        user_query="Look up C001.",
        category="tool_required",
        available_tools=["lookup"],
    )

    trace = run_episode(
        agent=ReplayAgent(
            action_sequence=[
                ToolCall("lookup", {}),
                ToolCall("lookup", {"customer_id": "C001"}),
                FinalAnswer("C001"),
            ]
        ),
        task=task,
        tool_defs={"lookup": tool},
        config=EpisodeConfig(validate_arguments=True),
    )

    assert trace.steps[0].observation.status == "invalid_arguments"
    assert trace.steps[1].observation.status == "success"
    assert trace.final_response == "C001"
