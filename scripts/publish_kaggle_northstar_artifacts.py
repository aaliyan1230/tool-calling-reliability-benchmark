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


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    return result.returncode, result.stdout or "", result.stderr or ""


def _copy_required_runs(repo_root: Path, stage_runs: Path, label_prefix: str) -> list[str]:
    expected_suffixes = ["base-ms", "ft-ms", "delta", "matrix"]
    copied = []
    for suffix in expected_suffixes:
        name = f"{label_prefix}-{suffix}"
        src = repo_root / "runs" / name
        dst = stage_runs / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required run directory: {src}")
        shutil.copytree(src, dst)
        copied.append(name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish north-star artifacts under runs/ to a Kaggle dataset.")
    parser.add_argument("--dataset-slug", required=True, help="Dataset slug only, e.g. tcrb-qwen25-3b-northstar-artifacts")
    parser.add_argument("--title", required=True, help="Kaggle dataset title")
    parser.add_argument("--label-prefix", required=True, help="North-star label prefix to publish")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument(
        "--stage-dir",
        default="tmp/kaggle_northstar_publish",
        help="Temporary staging directory",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _load_env_file(repo_root / ".env")

    if not os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

    username = os.environ.get("KAGGLE_USERNAME")
    if not username or not os.environ.get("KAGGLE_KEY"):
        raise RuntimeError("Missing KAGGLE_USERNAME/KAGGLE_KEY auth in env or .env.")

    stage_dir = (repo_root / args.stage_dir).resolve()
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage_runs = stage_dir / "runs"
    stage_runs.mkdir(parents=True, exist_ok=True)
    copied = _copy_required_runs(repo_root, stage_runs, args.label_prefix)

    metadata = {
        "title": args.title,
        "id": f"{username}/{args.dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (stage_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    create_cmd = ["kaggle", "datasets", "create", "-p", str(stage_dir), "-q"]
    rc, out, err = _run(create_cmd, cwd=repo_root)
    if rc == 0:
        print(out)
        print(json.dumps({"mode": "create", "dataset": metadata["id"], "copied": copied}, indent=2))
        return

    if "already exists" not in (out + err).lower():
        if out:
            print(out)
        if err:
            print(err)
        raise RuntimeError("Kaggle dataset create failed and did not indicate existing dataset.")

    version_cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(stage_dir),
        "-m",
        f"Update north-star artifacts for {args.label_prefix}",
        "-q",
    ]
    rc, out, err = _run(version_cmd, cwd=repo_root)
    if rc != 0:
        if out:
            print(out)
        if err:
            print(err)
        raise RuntimeError("Kaggle dataset version failed.")

    print(out)
    print(json.dumps({"mode": "version", "dataset": metadata["id"], "copied": copied}, indent=2))


if __name__ == "__main__":
    main()
