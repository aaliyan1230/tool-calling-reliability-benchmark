from tcrb.v02.agent import ReplayAgent
from tcrb.v02.executor import run_episode
from tcrb.v02.types import Clarify, TaskDef


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
