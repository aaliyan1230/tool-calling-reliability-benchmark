from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def _load_env_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    if result.stdout:
        print(result.stdout)


def _find_adapter_root(download_dir: Path) -> Path:
    direct = download_dir / "adapter"
    if direct.exists():
        return direct

    # Handle nested structure from zipped directories.
    matches = list(download_dir.glob("**/adapter"))
    for candidate in matches:
        if (candidate / "adapter_config.json").exists() and (candidate / "adapter_model.safetensors").exists():
            return candidate

    # Handle flat file layout.
    if (download_dir / "adapter_config.json").exists() and (download_dir / "adapter_model.safetensors").exists():
        return download_dir

    raise FileNotFoundError(
        "No adapter payload found in downloaded dataset. Expected adapter/ or flat adapter files."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and materialize a Kaggle adapter dataset.")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset ref, e.g. owner/slug")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing .env and outputs/ directory (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        default="tmp/kaggle_adapter_pull",
        help="Temporary local download directory",
    )
    parser.add_argument(
        "--target-dir",
        default="outputs/ft-notebook/final",
        help="Target adapter directory in this repo",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _load_env_file(repo_root / ".env")

    # Allow .env compatibility where key is stored as KAGGLE_API_TOKEN.
    if not os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("Missing KAGGLE_USERNAME/KAGGLE_KEY auth in env or .env.")

    download_dir = (repo_root / args.download_dir).resolve()
    target_dir = (repo_root / args.target_dir).resolve()

    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    _run([
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(download_dir),
        "--unzip",
        "-o",
        "-q",
    ])

    adapter_root = _find_adapter_root(download_dir)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_root, target_dir)

    metadata = {
        "dataset": args.dataset,
        "download_dir": str(download_dir),
        "adapter_root": str(adapter_root),
        "target_dir": str(target_dir),
    }
    print(json.dumps(metadata, indent=2))
    print("Adapter materialized successfully.")


if __name__ == "__main__":
    main()
