from tcrb.study_gate import StudyGateThresholds, evaluate_study_gates


def _multi_seed_payload(policy_rows: dict[str, dict[str, float]]) -> dict:
    rows = []
    for policy, metrics in policy_rows.items():
        rows.append(
            {
                "policy": policy,
                "metrics": {
                    "task_success_rate": {"mean": float(metrics["task_success_rate"])},
                    "invalid_tool_call_rate": {
                        "mean": float(metrics["invalid_tool_call_rate"])
                    },
                },
            }
        )
    return {
        "type": "multi_seed",
        "aggregate_policy_metrics": rows,
    }


def test_study_gate_fails_on_base_vs_ft_flatline():
    base = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.45,
                "invalid_tool_call_rate": 0.10,
            }
        }
    )
    ft = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.45,
                "invalid_tool_call_rate": 0.10,
            }
        }
    )

    payload = evaluate_study_gates(
        base_payload=base,
        finetuned_payload=ft,
        thresholds=StudyGateThresholds(flatline_epsilon=1e-4),
    )

    assert payload["verdict"] == "FAIL"
    assert payload["checks"][0]["name"] == "base_vs_ft_nonflatline"
    assert payload["checks"][0]["passed"] is False


def test_study_gate_passes_with_nonflat_base_vs_ft_signal():
    base = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.45,
                "invalid_tool_call_rate": 0.10,
            }
        }
    )
    ft = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.52,
                "invalid_tool_call_rate": 0.08,
            }
        }
    )

    payload = evaluate_study_gates(
        base_payload=base,
        finetuned_payload=ft,
        thresholds=StudyGateThresholds(flatline_epsilon=1e-4),
    )

    assert payload["verdict"] == "PASS"
    assert payload["checks"][0]["passed"] is True


def test_study_gate_fails_when_ft_matches_null_control_effect():
    base = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.45,
                "invalid_tool_call_rate": 0.10,
            }
        }
    )
    ft = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.52,
                "invalid_tool_call_rate": 0.08,
            }
        }
    )
    null_control = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.52,
                "invalid_tool_call_rate": 0.08,
            }
        }
    )

    payload = evaluate_study_gates(
        base_payload=base,
        finetuned_payload=ft,
        null_payload=null_control,
        thresholds=StudyGateThresholds(
            flatline_epsilon=1e-4,
            min_effect_vs_null=1e-3,
        ),
    )

    assert payload["verdict"] == "FAIL"
    check_names = [check["name"] for check in payload["checks"]]
    assert "ft_distinct_from_null_control" in check_names
    null_check = next(
        check for check in payload["checks"] if check["name"] == "ft_distinct_from_null_control"
    )
    assert null_check["passed"] is False


def test_study_gate_fails_when_matrix_signal_required_and_missing():
    base = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.45,
                "invalid_tool_call_rate": 0.10,
            }
        }
    )
    ft = _multi_seed_payload(
        {
            "naive_retry": {
                "task_success_rate": 0.52,
                "invalid_tool_call_rate": 0.08,
            }
        }
    )

    payload = evaluate_study_gates(
        base_payload=base,
        finetuned_payload=ft,
        thresholds=StudyGateThresholds(
            flatline_epsilon=1e-4,
            require_matrix_signal=True,
        ),
    )

    assert payload["verdict"] == "FAIL"
    check_names = [check["name"] for check in payload["checks"]]
    assert "matrix_input_required" in check_names