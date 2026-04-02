from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an Ollama model tag from a PEFT adapter directory"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Existing Ollama base model tag (for example qwen3:4b)",
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Path to PEFT adapter directory (must include adapter_config.json and adapter_model.safetensors)",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Target Ollama tag to create (for example qwen3:4b-ft)",
    )
    parser.add_argument(
        "--modelfile",
        default="models/ollama/Modelfile.adapter.generated",
        help="Path to write generated Modelfile",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Run 'ollama create' after writing Modelfile",
    )
    return parser.parse_args()


def _validate_adapter_dir(path: Path) -> None:
    required = [
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_text = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "adapter directory is missing required files:\n" + missing_text
        )


def _write_modelfile(*, base_model: str, adapter_dir: Path, modelfile: Path) -> None:
    modelfile.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"FROM {base_model}",
            f"ADAPTER {adapter_dir.resolve()}",
            "PARAMETER temperature 0",
            "",
        ]
    )
    modelfile.write_text(content, encoding="utf-8")


def _create_ollama_tag(*, tag: str, modelfile: Path) -> None:
    cmd = ["ollama", "create", tag, "-f", str(modelfile)]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr)
        raise RuntimeError(f"ollama create failed with code {completed.returncode}")


def main() -> int:
    args = _parse_args()
    adapter_dir = Path(args.adapter_dir).resolve()
    modelfile = Path(args.modelfile).resolve()

    _validate_adapter_dir(adapter_dir)
    _write_modelfile(
        base_model=str(args.base_model),
        adapter_dir=adapter_dir,
        modelfile=modelfile,
    )

    print(f"Wrote Modelfile: {modelfile}")
    print(f"Base model: {args.base_model}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"Target tag: {args.tag}")

    if args.create:
        _create_ollama_tag(tag=str(args.tag), modelfile=modelfile)
        print(f"Created Ollama tag: {args.tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
