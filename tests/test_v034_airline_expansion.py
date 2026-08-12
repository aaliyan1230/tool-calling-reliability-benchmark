from __future__ import annotations

from tcrb.v034.airline_expansion import _eligible


def _config() -> dict:
    return {
        "excluded_task_ids": ["1"],
        "development_seed_trajectory_ids": ["dev"],
        "min_event_count": 2,
    }


def _trace(**overrides: object) -> dict:
    value = {
        "trajectory_id": "new",
        "domain": "airline",
        "task_id": "2",
        "events": [{"event_id": "e1"}, {"event_id": "w", "tool_call": {"name": "update_reservation_flights"}}],
        "write_event_ids": ["w"],
    }
    value.update(overrides)
    return value


def test_airline_seed_eligibility_accepts_a_new_trace() -> None:
    assert _eligible(_trace(), _config(), set()) == []


def test_airline_seed_eligibility_rejects_shortlisted_development_and_malformed_traces() -> None:
    assert "task is in the excluded shortlist" in _eligible(_trace(task_id="1"), _config(), set())
    assert "trajectory is a held-out development seed" in _eligible(_trace(trajectory_id="dev"), _config(), set())
    assert "trajectory was already human-reviewed" in _eligible(_trace(), _config(), {"new"})
    assert "trace has no state-changing write" in _eligible(_trace(write_event_ids=[]), _config(), set())
    assert "trace is shorter than the configured minimum" in _eligible(_trace(events=[{"event_id": "e1"}]), _config(), set())
