from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_KEEP = (
    "src",
    "scripts",
    "configs",
    "workloads",
    "outputs/research/qwen25-3b-dpo-router-repair/dpo_train.jsonl",
    "kaggle",
    "pyproject.toml",
    "uv.lock",
    "README.md",
)


def run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("[kaggle-tcrb]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), text=True, check=False)


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "*.pyc",
            ),
        )
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_kernel(args: argparse.Namespace) -> Path:
    repo_root = Path.cwd()
    build_dir = Path(args.build_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    copy_path(repo_root / "kaggle", build_dir / "kaggle")
    runner_src = build_dir / "kaggle" / "runner.py"
    metadata_src = build_dir / "kaggle" / "kernel-metadata.json"
    shutil.copy2(runner_src, build_dir / "runner.py")

    metadata = json.loads(metadata_src.read_text(encoding="utf-8"))
    metadata["id"] = args.kernel_id
    metadata["title"] = args.title
    metadata["enable_gpu"] = bool(args.enable_gpu)
    metadata["enable_internet"] = bool(args.enable_internet)
    metadata["dataset_sources"] = [args.dataset_id]
    (build_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "kernel_id": args.kernel_id,
        "build_dir": str(build_dir),
        "dataset_id": args.dataset_id,
    }
    (build_dir / "tcrb_kaggle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[kaggle-tcrb] Prepared {build_dir}")
    return build_dir


def prepare_dataset(args: argparse.Namespace) -> Path:
    repo_root = Path.cwd()
    dataset_dir = Path(args.dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    for relative in DEFAULT_KEEP:
        if relative == "kaggle":
            continue
        copy_path(repo_root / relative, dataset_dir / relative)

    adapter_path = repo_root / args.adapter_path
    if adapter_path.exists() and not args.no_adapter:
        copy_path(adapter_path, dataset_dir / args.adapter_path)

    metadata = {
        "id": args.dataset_id,
        "title": args.dataset_title,
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    (dataset_dir / "dataset-metadata.json").write_text(
        metadata_text,
        encoding="utf-8",
    )
    (dataset_dir / "datasets-metadata.json").write_text(
        metadata_text,
        encoding="utf-8",
    )
    print(f"[kaggle-tcrb] Prepared dataset {dataset_dir}")
    return dataset_dir


def push_kernel(args: argparse.Namespace) -> int:
    build_dir = prepare_kernel(args)
    cmd = ["kaggle", "kernels", "push", "-p", str(build_dir)]
    if args.kernel_timeout:
        cmd.extend(["--timeout", str(args.kernel_timeout)])
    if args.accelerator:
        cmd.extend(["--accelerator", args.accelerator])
    return run(cmd, cwd=Path.cwd()).returncode


def dataset_create(args: argparse.Namespace) -> int:
    dataset_dir = prepare_dataset(args)
    cmd = ["kaggle", "datasets", "create", "-p", str(dataset_dir), "--dir-mode", "tar"]
    return run(cmd, cwd=Path.cwd()).returncode


def dataset_version(args: argparse.Namespace) -> int:
    dataset_dir = prepare_dataset(args)
    message = args.message or f"TCRB repo snapshot for {args.kernel_id}"
    cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(dataset_dir),
        "-m",
        message,
        "--dir-mode",
        "tar",
    ]
    return run(cmd, cwd=Path.cwd()).returncode


def status(args: argparse.Namespace) -> int:
    return run(["kaggle", "kernels", "status", args.kernel_id], cwd=Path.cwd()).returncode


def output(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "kernels",
        "output",
        args.kernel_id,
        "-p",
        str(output_dir),
        "-o",
    ]
    return run(cmd, cwd=Path.cwd()).returncode


def watch(args: argparse.Namespace) -> int:
    deadline = time.time() + float(args.timeout_seconds)
    while time.time() < deadline:
        completed = subprocess.run(
            ["kaggle", "kernels", "status", args.kernel_id],
            cwd=str(Path.cwd()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output_text = completed.stdout.strip()
        print(output_text, flush=True)
        lowered = output_text.lower()
        if any(
            token in lowered
            for token in ("complete", "error", "failed", "cancel", "stopped")
        ):
            return completed.returncode
        time.sleep(float(args.poll_seconds))
    raise SystemExit(f"timed out waiting for {args.kernel_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, push, poll, and download TCRB Kaggle experiment kernels"
    )
    parser.add_argument(
        "--kernel-id",
        default="aaliyanshaikh/tcrb-hf-signal-run",
        help="Kaggle kernel id as owner/slug",
    )
    parser.add_argument("--title", default="TCRB HF Signal Run")
    parser.add_argument("--build-dir", default="tmp/kaggle_tcrb_kernel")
    parser.add_argument("--dataset-id", default="aaliyanshaikh/tcrb-repo-snapshot")
    parser.add_argument("--dataset-title", default="TCRB Repo Snapshot")
    parser.add_argument("--dataset-dir", default="tmp/kaggle_tcrb_dataset")
    parser.add_argument("--adapter-path", default="outputs/ft-notebook/final")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--enable-gpu", action="store_true", default=True)
    parser.add_argument("--enable-internet", action="store_true", default=True)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument(
        "--kernel-timeout",
        type=int,
        default=1800,
        help="Kaggle run timeout passed to kernels push",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Stage a Kaggle kernel directory")
    subparsers.add_parser("prepare-dataset", help="Stage a Kaggle dataset directory")
    subparsers.add_parser("dataset-create", help="Create the private source dataset")
    version_parser = subparsers.add_parser(
        "dataset-version", help="Upload a new source dataset version"
    )
    version_parser.add_argument("--message", default=None)
    subparsers.add_parser("push", help="Stage and push the Kaggle kernel")
    subparsers.add_parser("status", help="Show latest kernel status")

    watch_parser = subparsers.add_parser("watch", help="Poll until completion/failure")
    watch_parser.add_argument("--poll-seconds", type=float, default=60.0)
    watch_parser.add_argument("--timeout-seconds", type=float, default=21600.0)

    output_parser = subparsers.add_parser("output", help="Download latest kernel output")
    output_parser.add_argument(
        "--output-dir",
        default="tmp/kaggle_tcrb_output",
        help="Directory to download kernel outputs into",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_kernel(args)
        return 0
    if args.command == "prepare-dataset":
        prepare_dataset(args)
        return 0
    if args.command == "dataset-create":
        return dataset_create(args)
    if args.command == "dataset-version":
        return dataset_version(args)
    if args.command == "push":
        return push_kernel(args)
    if args.command == "status":
        return status(args)
    if args.command == "watch":
        return watch(args)
    if args.command == "output":
        return output(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
