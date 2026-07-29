import importlib.util
from pathlib import Path

from tcrb.v02.agent import RECOVERY_SYSTEM_PROMPT


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_v02_recovery_dataset.py"
_SPEC = importlib.util.spec_from_file_location("v02_recovery_dataset", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
build_recovery_rows = _MODULE.build_recovery_rows


def test_recovery_dataset_prefers_retry_after_a_failed_tool_call():
    traces = [
        {
            "task_id": "TEST-001",
            "split": "train",
            "hazard": "execution_error",
            "fault_applied": True,
            "success": True,
            "canonical_claims": ["C001"],
            "final_response": "Task completed for C001.",
            "task_query": "Look up customer C001.",
            "available_tools": ["customer_lookup"],
            "steps": [
                {
                    "step_index": 0,
                    "action": {
                        "type": "tool_call",
                        "name": "customer_lookup",
                        "arguments": {"customer_id": "C001"},
                    },
                    "observation": {
                        "status": "execution_error",
                        "payload": {"error": "temporary"},
                    },
                },
                {
                    "step_index": 1,
                    "action": {"type": "final_answer", "text": "Task completed for C001."},
                },
            ],
        }
    ]

    sft_rows, dpo_rows = build_recovery_rows(traces)

    assert len(sft_rows) == 1
    assert len(dpo_rows) == 1
    assert RECOVERY_SYSTEM_PROMPT in sft_rows[0]["text"]
    assert '"name": "customer_lookup"' in sft_rows[0]["text"]
    assert sft_rows[0]["messages"][-1]["content"] == '{"final_answer": "Task completed for C001."}'
    assert dpo_rows[0]["chosen"] == '{"final_answer": "Task completed for C001."}'
    assert dpo_rows[0]["rejected"] == '{"name": "customer_lookup", "arguments": {"customer_id": "C001"}}'


def test_recovery_dataset_skips_traces_without_failed_tool_steps():
    sft_rows, dpo_rows = build_recovery_rows(
        [
            {
                "task_id": "TEST-002",
                "split": "train",
                "hazard": "execution_error",
                "task_query": "Answer this.",
                "available_tools": [],
                "steps": [
                    {
                        "step_index": 0,
                        "action": {"type": "final_answer", "text": "Answer"},
                    }
                ],
            }
        ]
    )

    assert sft_rows == []
    assert dpo_rows == []


def test_recovery_dataset_excludes_heldout_split_and_hazard_traces():
    traces = [
        {
            "task_id": "TEST-003",
            "split": "test",
            "hazard": "execution_error",
            "fault_applied": True,
            "steps": [],
        },
        {
            "task_id": "TEST-004",
            "split": "train",
            "hazard": "silent_corruption",
            "fault_applied": True,
            "steps": [],
        },
    ]

    sft_rows, dpo_rows = build_recovery_rows(traces)

    assert sft_rows == []
    assert dpo_rows == []
