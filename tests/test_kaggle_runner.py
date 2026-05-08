from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "kaggle" / "runner.py"
    spec = importlib.util.spec_from_file_location("tcrb_kaggle_runner", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_applies_extra_env_and_pythonpath(monkeypatch, tmp_path: Path):
    runner = _load_runner_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(runner, "REPO_ROOT", repo_root)
    monkeypatch.setattr(runner, "LOG_DIR", log_dir)

    captured: dict[str, object] = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = dict(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)

    runner.run(["python", "-V"], log_name="test.log", extra_env={"CUDA_VISIBLE_DEVICES": "0"})

    assert captured["cmd"] == ["python", "-V"]
    assert captured["cwd"] == str(repo_root)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["env"]["PYTHONPATH"].startswith(str(repo_root / "src"))
