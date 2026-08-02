import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_v02_oracle_recovery_dataset.py"
_SPEC = importlib.util.spec_from_file_location("v02_oracle_recovery_dataset", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_oracle_recovery_rows_cover_seen_training_hazards_without_leakage():
    sft_rows, dpo_rows = _MODULE.build_oracle_recovery_rows()

    assert len(sft_rows) == len(dpo_rows)
    assert len(sft_rows) > 0
    assert {row["metadata"]["hazard"] for row in sft_rows} == {
        "execution_error",
        "schema_drift",
        "partial_output",
    }
    assert all(row["metadata"]["split"] == "train" for row in sft_rows)
    assert all(row["metadata"]["domain"] in {"customer_support", "ecommerce"} for row in sft_rows)

    for sft_row, dpo_row in zip(sft_rows, dpo_rows):
        chosen = json.loads(sft_row["messages"][-1]["content"])
        assert "name" in chosen
        assert chosen["name"] == sft_row["metadata"]["failed_tool"]
        assert "Available tools:" in sft_row["messages"][0]["content"]
        assert sft_row["messages"][3]["role"] == "user"
        assert sft_row["messages"][3]["content"].startswith("Tool result: ")
        assert dpo_row["chosen"] == sft_row["messages"][-1]["content"]
        assert dpo_row["chosen"] != dpo_row["rejected"]
