from tcrb.v02.eval_runner import _episode_success
from tcrb.v02.types import EpisodeTrace


def test_episode_success_requires_canonical_claims_when_present():
    trace = EpisodeTrace(
        task_id="DEVE-010",
        domain="developer_tools",
        success=True,
        final_response="Task completed.",
    )

    assert _episode_success(trace, ["BUILD-402"]) is False


def test_episode_success_accepts_a_final_answer_for_claimless_tasks():
    trace = EpisodeTrace(
        task_id="DEVE-031",
        domain="developer_tools",
        success=True,
        final_response="Task completed.",
    )

    assert _episode_success(trace, []) is True
