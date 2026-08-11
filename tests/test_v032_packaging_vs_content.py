"""Tests for the v3.2 2x2.

The whole run rests on two claims: the "values kept" views really keep every
answer-critical value, and the "values stripped" views really keep none of them.
If either slips, the factor labels are lies and every number downstream is
meaningless. Most of what follows checks exactly that.
"""

from __future__ import annotations

import json

import pytest

from tcrb.v031.cases import build_tcrb_traces
from tcrb.v031.views import build_view as v31_build_view
from tcrb.v032.schema import CONTENT, PACKAGING, VIEW_TYPES
from tcrb.v032.views import (
    build_all_views,
    build_view,
    critical_values_by_event,
    exposed_values_by_event,
    validate_views,
)


@pytest.fixture(scope="module")
def traces():
    return build_tcrb_traces()


@pytest.fixture(scope="module")
def pairs(traces):
    return build_all_views(traces, VIEW_TYPES)


def test_dataset_shape(traces, pairs):
    assert len(traces) == 24
    assert len(pairs) == len(traces) * 4
    outcomes = [trace.gold_outcome for trace in traces]
    assert outcomes.count("safe") == outcomes.count("unsafe") == 12


def test_validation_is_clean(traces, pairs):
    assert validate_views(traces, pairs) == []


def test_values_kept_views_keep_every_critical_value(traces):
    for trace in traces:
        needed = critical_values_by_event(trace)
        for view_type in ("full", "lossless_summary"):
            view, _ = build_view(trace, view_type)
            exposed = exposed_values_by_event(view)
            for event_id, values in needed.items():
                missing = values - exposed.get(event_id, set())
                assert not missing, (
                    f"{trace.task_family}/{view_type}/{event_id} lost {sorted(missing)}"
                )


def test_values_stripped_views_keep_none_of_them(traces):
    for trace in traces:
        needed = critical_values_by_event(trace)
        for view_type in ("raw_projection", "actionable_summary"):
            view, _ = build_view(trace, view_type)
            exposed = exposed_values_by_event(view)
            for event_id in needed:
                seen = exposed.get(event_id, set())
                assert not seen, (
                    f"{trace.task_family}/{view_type}/{event_id} leaked {sorted(seen)}"
                )


def test_every_trace_has_something_to_strip(traces):
    """A trace with no answer-critical values would sit in both columns at once."""
    for trace in traces:
        needed = critical_values_by_event(trace)
        assert needed, f"{trace.task_family} has no required events"
        assert any(values for values in needed.values()), (
            f"{trace.task_family} has no answer-critical values to strip"
        )


def test_packaging_label_matches_presence_of_a_summariser(traces):
    for trace in traces:
        for view_type in VIEW_TYPES:
            view, _ = build_view(trace, view_type)
            wrapped = any(
                event.get("source") == "handoff_summarizer" for event in view.events
            )
            assert wrapped == (PACKAGING[view_type] == "summary"), view_type


def test_lossless_and_actionable_differ_only_by_payload(traces):
    """The cleanest one-factor comparison in the design.

    Same summariser, same items, same order. The answer-critical items carry a
    `payload` in one and not the other, and nothing else moves.
    """
    for trace in traces:
        lossless, _ = build_view(trace, "lossless_summary")
        actionable, _ = build_view(trace, "actionable_summary")

        left = _summary_items(lossless)
        right = _summary_items(actionable)
        assert [item["event_id"] for item in left] == [
            item["event_id"] for item in right
        ]
        for lhs, rhs in zip(left, right):
            assert {k: v for k, v in lhs.items() if k != "payload"} == {
                k: v for k, v in rhs.items() if k != "payload"
            }


def test_raw_projection_strips_the_same_fields_actionable_does(traces):
    """The two stripped cells must differ in packaging only, not in content.

    Compared per event, because the two views describe the same events in
    different shapes: raw events in one, summariser items in the other.
    """
    for trace in traces:
        raw, _ = build_view(trace, "raw_projection")
        actionable, _ = build_view(trace, "actionable_summary")
        needed = critical_values_by_event(trace)
        raw_exposed = exposed_values_by_event(raw)
        act_exposed = exposed_values_by_event(actionable)
        for event_id in needed:
            assert raw_exposed.get(event_id, set()) == act_exposed.get(
                event_id, set()
            ), f"{trace.task_family}/{event_id}"

        assert _field_names(raw, needed) == _field_names(actionable, needed)


def test_raw_projection_has_no_summariser_anywhere(traces):
    for trace in traces:
        view, _ = build_view(trace, "raw_projection")
        blob = json.dumps(list(view.events))
        assert "handoff_summarizer" not in blob
        assert "compression_policy" not in blob


def test_carried_over_views_are_identical_to_v31(traces):
    """Otherwise v3.2 cannot claim to replicate v3.1."""
    for trace in traces:
        for view_type in ("full", "actionable_summary"):
            old, _ = v31_build_view(trace, view_type)
            new, _ = build_view(trace, view_type)
            assert json.dumps(list(old.events), sort_keys=True) == json.dumps(
                list(new.events), sort_keys=True
            ), f"{trace.task_family}/{view_type} drifted from v3.1"


def test_views_never_leak_private_labels(traces, pairs):
    forbidden = {
        "gold_outcome",
        "required_evidence_ids",
        "mutation_id",
        "view_type",
        "faults_applied",
    }
    for view, _ in pairs:
        blob = json.dumps(view.to_public_dict(), sort_keys=True)
        for key in forbidden:
            assert f'"{key}"' not in blob, f"{view.view_id} leaked {key}"


def test_view_ids_are_unique_and_disjoint_from_v31(traces, pairs):
    ids = [view.view_id for view, _ in pairs]
    assert len(ids) == len(set(ids))
    v31_ids = {
        v31_build_view(trace, view_type)[0].view_id
        for trace in traces
        for view_type in ("full", "actionable_summary")
    }
    assert not (set(ids) & v31_ids)


def test_gold_stays_paired_with_its_view(pairs):
    for view, gold in pairs:
        assert gold.view_id == view.view_id
        assert gold.view_type == view.view_type
        assert gold.trace_id == view.trace_id


def test_subject_report_is_present_in_every_view(pairs):
    """The claim under review must always be visible, or the task is impossible."""
    for view, _ in pairs:
        assert any(event.get("event_id") == "subject_output" for event in view.events)


def test_factor_table_covers_every_view_type():
    assert set(PACKAGING) == set(CONTENT) == set(VIEW_TYPES)
    assert sorted({(PACKAGING[v], CONTENT[v]) for v in VIEW_TYPES}) == [
        ("raw", "values_present"),
        ("raw", "values_stripped"),
        ("summary", "values_present"),
        ("summary", "values_stripped"),
    ]


def _field_names(view, needed: dict) -> dict[str, list[str]]:
    """Which field names each required event advertises, however it is packaged."""
    found: dict[str, list[str]] = {}
    for event in view.events:
        if event.get("source") == "handoff_summarizer":
            for item in event["payload"]["summary"]:
                if item["event_id"] in needed:
                    found[item["event_id"]] = sorted(item.get("available_fields", []))
        elif event["event_id"] in needed:
            found[event["event_id"]] = sorted(
                event.get("payload", {}).get("available_fields", [])
            )
    return found


def _summary_items(view) -> list[dict]:
    for event in view.events:
        if event.get("source") == "handoff_summarizer":
            return list(event["payload"]["summary"])
    raise AssertionError("no summary event found")
