from tcrb.v02.metrics import count_diagnostic_labels, is_recovery


def _fault(success: bool, applied: bool) -> dict:
    return {"success": success, "fault_applied": applied}


def _record(clean_labels: list[str], faulted: list[dict]) -> dict:
    return {"clean": {"diagnostic_labels": clean_labels}, "faulted": faulted}


def test_is_recovery_requires_success_and_applied_fault():
    assert is_recovery(_fault(True, True)) is True
    assert is_recovery(_fault(True, False)) is False
    assert is_recovery(_fault(False, True)) is False
    assert is_recovery(_fault(False, False)) is False


def test_count_diagnostic_labels_includes_clean_and_faulted():
    results = [
        _record(["tool_skip"], [{"diagnostic_labels": ["retry_loop"]}]),
        _record([], [{"diagnostic_labels": ["retry_loop", "tool_skip"]}]),
        _record(["result_ignore"], []),
    ]
    assert count_diagnostic_labels(results) == {
        "tool_skip": 2,
        "retry_loop": 2,
        "result_ignore": 1,
    }


def test_count_diagnostic_labels_handles_clean_only_and_empty():
    assert count_diagnostic_labels([]) == {}
    assert count_diagnostic_labels([_record(["tool_skip"], [])]) == {"tool_skip": 1}
