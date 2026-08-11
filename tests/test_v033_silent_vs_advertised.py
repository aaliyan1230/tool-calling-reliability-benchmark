"""Tests for the v3.3 silent-vs-advertised 2x2.

Two properties carry the whole experiment:

1. A *silent* view must contain nothing that hints evidence was removed. If any
   gap signal survives, the silent arm is just a rerun of v3.2.
2. After a drop, the safe and unsafe twins must present **identical evidence**.
   That is what makes the asymmetric cells a real test: the monitor cannot tell
   the two apart from the evidence, only from the claim.

Most of what follows checks those two.
"""

from __future__ import annotations

import json

import pytest

from tcrb.v031.cases import build_tcrb_traces
from tcrb.v031.views import build_view as v31_build_view
from tcrb.v033.schema import ANNOUNCEMENT, DEGRADED_VIEW_TYPES, SCOPE, VIEW_TYPES
from tcrb.v033.views import (
    CONTRADICTION_FIELDS,
    build_all_views,
    build_view,
    drop_map,
    evidence_blob,
    records_by_event,
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
    assert len(pairs) == len(traces) * 5


def test_validation_is_clean(traces, pairs):
    assert validate_views(traces, pairs) == []


def test_contradiction_table_covers_every_family(traces):
    assert {t.task_family for t in traces} == set(CONTRADICTION_FIELDS)


def test_contradiction_fields_exist_on_their_events(traces):
    """A typo in the table would silently drop nothing at all."""
    for trace in traces:
        table = CONTRADICTION_FIELDS[trace.task_family]
        by_id = {e.event_id: e for e in trace.events}
        for event_id, fields in table.items():
            assert event_id in by_id, f"{trace.task_family}: no event {event_id}"
            assert event_id in trace.required_evidence_ids, (
                f"{trace.task_family}: {event_id} is not answer-critical"
            )
            for field in fields:
                assert field in by_id[event_id].payload, (
                    f"{trace.task_family}: {event_id} has no field {field}"
                )


def test_silent_views_carry_no_gap_signal(traces):
    """The load-bearing property of the silent arm."""
    for trace in traces:
        for view_type in DEGRADED_VIEW_TYPES:
            if ANNOUNCEMENT[view_type] != "silent":
                continue
            view, _ = build_view(trace, view_type)
            blob = json.dumps(view.to_public_dict(), sort_keys=True)
            for signal in ("available_fields", "observed", "redacted", "projection", "missing"):
                assert signal not in blob, f"{trace.task_family}/{view_type} leaks {signal}"


def test_advertised_views_do_name_their_gap(traces):
    for trace in traces:
        for view_type in DEGRADED_VIEW_TYPES:
            if ANNOUNCEMENT[view_type] != "advertised":
                continue
            view, _ = build_view(trace, view_type)
            blob = json.dumps(view.to_public_dict(), sort_keys=True)
            assert "available_fields" in blob, f"{trace.task_family}/{view_type}"


def test_twins_show_identical_evidence_in_every_degraded_view(traces):
    """The invariant that makes the asymmetric cells a fair test.

    If safe and unsafe present the same evidence, any difference in verdict has
    to come from the claim itself, not from something the monitor spotted.
    """
    by_family = {}
    for trace in traces:
        for view_type in DEGRADED_VIEW_TYPES:
            view, _ = build_view(trace, view_type)
            by_family.setdefault((trace.task_family, view_type), {})[
                trace.gold_outcome
            ] = view
    for (family, view_type), outcomes in by_family.items():
        assert evidence_blob(outcomes["safe"]) == evidence_blob(outcomes["unsafe"]), (
            f"{family}/{view_type}: twins differ, the discriminating fact survived"
        )


def test_full_view_twins_do_differ(traces):
    """Sanity check on the check: on the complete trace the twins are
    distinguishable, otherwise the invariant above would be vacuous."""
    subject = {}
    for trace in traces:
        view, _ = build_view(trace, "full")
        subject.setdefault(trace.task_family, {})[trace.gold_outcome] = json.dumps(
            [e for e in view.events if e["event_id"] == "subject_output"], sort_keys=True
        )
    for family, outcomes in subject.items():
        assert outcomes["safe"] != outcomes["unsafe"], family


def test_every_family_loses_something_under_both_scopes(traces):
    for trace in traces:
        for scope in ("symmetric", "asymmetric"):
            dropped = drop_map(trace, scope)
            assert any(dropped.values()), f"{trace.task_family}/{scope} dropped nothing"


def test_symmetric_drops_at_least_as_much_as_asymmetric(traces):
    """Symmetric is defined as the union, so it can never drop less."""
    for trace in traces:
        sym = drop_map(trace, "symmetric")
        asym = drop_map(trace, "asymmetric")
        for event_id, fields in asym.items():
            assert fields <= sym.get(event_id, set()), f"{trace.task_family}/{event_id}"


def test_cs_return_is_actually_bitten(traces):
    """v3.2's blind spot in the *instrument*: this family is pure identifier
    linkage, and a rule that spared identifiers left it untouched."""
    for trace in traces:
        if trace.task_family != "cs_return_r5003_owner":
            continue
        for scope in ("symmetric", "asymmetric"):
            dropped = drop_map(trace, scope)
            assert "customer_id" in dropped.get("order_record", set()), scope
            assert "customer_id" in dropped.get("customer_record", set()), scope


def test_dropped_fields_are_gone_from_their_own_records(traces):
    for trace in traces:
        for view_type in DEGRADED_VIEW_TYPES:
            view, _ = build_view(trace, view_type)
            records = records_by_event(view)
            for event_id, fields in drop_map(trace, SCOPE[view_type]).items():
                payload = records.get(event_id, {})
                for field in fields:
                    assert field not in payload, (
                        f"{trace.task_family}/{view_type}: {event_id}.{field} survived"
                    )


def test_asymmetric_keeps_the_agreeing_values(traces):
    """Asymmetric is only meaningful if something survives. A view that dropped
    everything would just be the symmetric one under a different name."""
    kept_somewhere = False
    for trace in traces:
        view, _ = build_view(trace, "asymmetric_silent")
        records = records_by_event(view)
        payloads = {e.event_id: e.payload for e in trace.events}
        for event_id in trace.required_evidence_ids:
            surviving = set(records.get(event_id, {}))
            dropped = drop_map(trace, "asymmetric").get(event_id, set())
            expected = set(payloads[event_id]) - dropped
            assert surviving == expected, f"{trace.task_family}/{event_id}"
            if surviving:
                kept_somewhere = True
    assert kept_somewhere


def test_full_view_is_identical_to_v31(traces):
    for trace in traces:
        old, _ = v31_build_view(trace, "full")
        new, _ = build_view(trace, "full")
        assert json.dumps(list(old.events), sort_keys=True) == json.dumps(
            list(new.events), sort_keys=True
        ), trace.task_family


def test_views_never_leak_private_labels(pairs):
    forbidden = {"gold_outcome", "required_evidence_ids", "mutation_id", "view_type"}
    for view, _ in pairs:
        blob = json.dumps(view.to_public_dict(), sort_keys=True)
        for key in forbidden:
            assert f'"{key}"' not in blob, f"{view.view_id} leaked {key}"


def test_subject_report_is_always_visible(pairs):
    for view, _ in pairs:
        assert any(e.get("event_id") == "subject_output" for e in view.events)


def test_view_ids_unique_and_disjoint_from_earlier_pilots(traces, pairs):
    ids = [v.view_id for v, _ in pairs]
    assert len(ids) == len(set(ids))
    v31_ids = {v31_build_view(t, "full")[0].view_id for t in traces}
    assert not (set(ids) & v31_ids)


def test_factor_tables_are_consistent():
    assert set(SCOPE) == set(ANNOUNCEMENT) == set(DEGRADED_VIEW_TYPES)
    assert sorted({(SCOPE[v], ANNOUNCEMENT[v]) for v in DEGRADED_VIEW_TYPES}) == [
        ("asymmetric", "advertised"),
        ("asymmetric", "silent"),
        ("symmetric", "advertised"),
        ("symmetric", "silent"),
    ]
