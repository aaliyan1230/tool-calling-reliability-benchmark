from tcrb.v02.eval_runner import _episode_success
from tcrb.v02.types import Clarify, EpisodeTrace, FinalAnswer, StepRecord, TaskDef, ToolCall


def _task(category, claims=None, sequences=None):
    return TaskDef(
        task_id="TEST-001",
        domain="test",
        user_query="Test query",
        category=category,
        available_tools=["lookup"],
        canonical_claims=claims or [],
        valid_tool_sequences=sequences or [],
    )


def test_episode_success_requires_canonical_claims_when_present():
    trace = EpisodeTrace(
        task_id="DEVE-010",
        domain="developer_tools",
        success=True,
        final_response="Task completed.",
    )

    assert _episode_success(trace, _task("tool_required", ["BUILD-402"], [["lookup"]])) is False


def test_episode_success_accepts_a_final_answer_for_claimless_tasks():
    trace = EpisodeTrace(
        task_id="DEVE-031",
        domain="developer_tools",
        success=True,
        final_response="Task completed.",
    )

    trace.steps = [
        StepRecord(0, "", FinalAnswer("Task completed."), None, None)
    ]

    assert _episode_success(trace, _task("no_tool")) is True


def test_no_tool_success_rejects_tool_use():
    trace = EpisodeTrace(
        task_id="TEST-001",
        domain="test",
        success=True,
        final_response="Paris",
        steps=[
            StepRecord(0, "", ToolCall("lookup", {}), None, None),
            StepRecord(1, "", FinalAnswer("Paris"), None, None),
        ],
    )

    assert _episode_success(trace, _task("no_tool", ["Paris"])) is False


def test_missing_information_success_requires_clarification():
    clarify_trace = EpisodeTrace(
        task_id="TEST-001",
        domain="test",
        steps=[StepRecord(0, "", Clarify("Which account?"), None, None)],
    )
    answer_trace = EpisodeTrace(
        task_id="TEST-001",
        domain="test",
        success=True,
        final_response="Task completed.",
        steps=[StepRecord(0, "", FinalAnswer("Task completed."), None, None)],
    )

    task = _task("missing_information")
    assert _episode_success(clarify_trace, task) is True
    assert _episode_success(answer_trace, task) is False


def test_tool_required_success_requires_valid_sequence_and_allows_retries():
    trace = EpisodeTrace(
        task_id="TEST-001",
        domain="test",
        success=True,
        final_response="Alice",
        steps=[
            StepRecord(0, "", ToolCall("lookup", {}), None, None),
            StepRecord(1, "", ToolCall("lookup", {}), None, None),
            StepRecord(2, "", FinalAnswer("Alice"), None, None),
        ],
    )

    assert _episode_success(trace, _task("tool_required", ["Alice"], [["lookup"]])) is True
