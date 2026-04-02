from tcrb.transfer_matrix import (
    MatrixThresholds,
    gate_eval_case_delta,
    summarize_eval_case_score,
)


def test_summarize_eval_case_score_weighted_mean():
    payload = {
        "cases_total": 4,
        "policies": [
            {"cases_evaluated": 3, "first_tool_accuracy": 1.0, "sequence_prefix_accuracy": 0.5},
            {"cases_evaluated": 1, "first_tool_accuracy": 0.0, "sequence_prefix_accuracy": 1.0},
        ],
    }

    summary = summarize_eval_case_score(payload)
    assert summary.cases_total == 4
    assert abs(summary.first_tool_accuracy - 0.75) < 1e-9
    assert abs(summary.sequence_prefix_accuracy - 0.625) < 1e-9


def test_gate_eval_case_delta_target_thresholds():
    thresholds = MatrixThresholds(
        target_first_tool_min_delta=0.02,
        target_sequence_min_delta=0.02,
        open_first_tool_min_delta=-0.03,
        open_sequence_min_delta=-0.03,
    )

    assert (
        gate_eval_case_delta(
            split="target",
            first_tool_delta=0.03,
            sequence_delta=0.02,
            thresholds=thresholds,
        )
        == "PASS"
    )
    assert (
        gate_eval_case_delta(
            split="target",
            first_tool_delta=0.03,
            sequence_delta=0.0,
            thresholds=thresholds,
        )
        == "HOLD"
    )
    assert (
        gate_eval_case_delta(
            split="target",
            first_tool_delta=-0.01,
            sequence_delta=0.0,
            thresholds=thresholds,
        )
        == "FAIL"
    )


def test_gate_eval_case_delta_open_thresholds():
    thresholds = MatrixThresholds(
        target_first_tool_min_delta=0.02,
        target_sequence_min_delta=0.02,
        open_first_tool_min_delta=-0.02,
        open_sequence_min_delta=-0.01,
    )

    assert (
        gate_eval_case_delta(
            split="open",
            first_tool_delta=-0.01,
            sequence_delta=-0.01,
            thresholds=thresholds,
        )
        == "PASS"
    )
    assert (
        gate_eval_case_delta(
            split="open",
            first_tool_delta=-0.03,
            sequence_delta=-0.01,
            thresholds=thresholds,
        )
        == "HOLD"
    )
