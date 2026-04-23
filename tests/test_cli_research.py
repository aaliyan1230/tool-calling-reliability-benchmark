import json
import sys
from pathlib import Path

from tcrb.cli import main


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_prepare_sft_data_cli_writes_masked_jsonl(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "sharegpt.json"
    recipe_path = tmp_path / "recipe.json"
    output_path = tmp_path / "prepared" / "sft_train.jsonl"

    _write_json(
        source_path,
        [
            {
                "system": "You can call tools.",
                "conversations": [
                    {"from": "human", "value": "Weather in Berlin?"},
                    {
                        "from": "gpt",
                        "value": '{"name":"weather.lookup","arguments":{"city":"Berlin"}}',
                    },
                ],
                "tools": [
                    {"type": "function", "function": {"name": "weather.lookup"}}
                ],
            }
        ],
    )
    _write_json(
        recipe_path,
        {
            "stage": "sft",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "output_dir": str(tmp_path / "outputs"),
            "dataset_sources": [
                {
                    "name": "toolace_local",
                    "format": "sharegpt",
                    "path": str(source_path),
                }
            ],
            "masking": {"ratio": 1.0, "seed": 3},
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tcrb",
            "prepare-sft-data",
            "--recipe-config",
            str(recipe_path),
            "--output-jsonl",
            str(output_path),
        ],
    )

    assert main() == 0

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["metadata"]["masking"]["replacements"]["weather.lookup"].startswith(
        "[MASK_FUNC_"
    )
    assert row["source"] == "toolace_local"


def test_mine_failure_pairs_cli_writes_preference_jsonl(tmp_path: Path, monkeypatch):
    eval_path = tmp_path / "eval.json"
    output_path = tmp_path / "mined.jsonl"

    _write_json(
        eval_path,
        {
            "cases": [
                {
                    "task_id": "t1",
                    "prompt": "Find the Berlin weather",
                    "expected_output": {
                        "name": "weather.lookup",
                        "arguments": {"city": "Berlin"},
                    },
                    "predicted_output": {
                        "name": "weather.find",
                        "arguments": {"city": "Berlin"},
                    },
                }
            ]
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tcrb",
            "mine-failure-pairs",
            "--eval-json",
            str(eval_path),
            "--output-jsonl",
            str(output_path),
            "--allowed-tools",
            "weather.lookup,weather.find",
        ],
    )

    assert main() == 0

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["failure_type"] == "wrong_function_name"
    assert row["metadata"]["task_id"] == "t1"