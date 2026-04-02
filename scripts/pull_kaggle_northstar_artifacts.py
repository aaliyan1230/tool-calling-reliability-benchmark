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


def _find_runs_root(download_dir: Path) -> Path:
    direct = download_dir / "runs"
    if direct.exists() and direct.is_dir():
        return direct

    # Some datasets are published with run directories at the archive root
    # (e.g., <label>-base-ms, <label>-ft-ms, <label>-delta, <label>-matrix).
    top_level_dirs = [p for p in download_dir.iterdir() if p.is_dir()]
    if top_level_dirs and any(name.name.endswith("-base-ms") for name in top_level_dirs):
        return download_dir

    matches = [p for p in download_dir.glob("**/runs") if p.is_dir()]
    if matches:
        # Prefer shallowest candidate if nested zip structure is present.
        return sorted(matches, key=lambda p: len(p.parts))[0]

    raise FileNotFoundError("No runs/ directory found in downloaded dataset payload.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset containing north-star run artifacts and materialize into runs/."
    )
    parser.add_argument("--dataset", required=True, help="Kaggle dataset ref, e.g. owner/slug")
    parser.add_argument(
        "--label-prefix",
        required=True,
        help="North-star label prefix, e.g. northstar-hf-kaggle-qwen25-3b",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing .env and runs/ directory (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        default="tmp/kaggle_northstar_pull",
        help="Temporary local download directory",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _load_env_file(repo_root / ".env")

    if not os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("Missing KAGGLE_USERNAME/KAGGLE_KEY auth in env or .env.")

    download_dir = (repo_root / args.download_dir).resolve()
    runs_root = (repo_root / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

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

    source_runs = _find_runs_root(download_dir)
    expected_suffixes = ["base-ms", "ft-ms", "delta", "matrix"]

    copied = []
    missing = []
    for suffix in expected_suffixes:
        name = f"{args.label_prefix}-{suffix}"
        src_dir = source_runs / name
        dst_dir = runs_root / name
        if not src_dir.exists():
            missing.append(name)
            continue
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        copied.append(str(dst_dir.relative_to(repo_root)))

    metadata = {
        "dataset": args.dataset,
        "label_prefix": args.label_prefix,
        "source_runs_root": str(source_runs),
        "copied": copied,
        "missing": missing,
    }
    print(json.dumps(metadata, indent=2))

    if missing:
        raise RuntimeError(f"Dataset downloaded, but missing expected runs: {missing}")

    print("North-star artifacts materialized successfully.")


if __name__ == "__main__":
    main()
