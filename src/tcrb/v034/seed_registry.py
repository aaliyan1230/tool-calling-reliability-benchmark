"""Validated seed registries for the scalable augmentation run."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .util import CONFIG_ROOT, DEFAULT_LOCAL_ROOT, content_id, read_json, read_jsonl, sha256_file, write_json, write_jsonl


def _trajectories(local_root: Path) -> dict[str, dict[str, Any]]:
    return {row["trajectory_id"]: row for row in read_jsonl(local_root / "normalized" / "trajectories.jsonl")}


def _merged_safe(local_root: Path) -> set[str]:
    return {row["annotation_id"] for row in read_jsonl(local_root / "annotations" / "merged.jsonl") if row.get("final_label") == "safe"}


def select_retail_seeds(local_root: Path = DEFAULT_LOCAL_ROOT, config_path: Path | None = None) -> dict[str, Any]:
    """Freeze 15 already two-reviewer-validated, task-diverse retail seeds."""
    config = read_json(config_path or CONFIG_ROOT / "retail_seed_expansion.json")
    trajectories = _trajectories(local_root)
    merged_safe = _merged_safe(local_root)
    selected = config["selected_seed_trajectory_ids"]
    if len(selected) != int(config["seed_count"]) or len(set(selected)) != len(selected):
        raise ValueError("retail seed config must contain 15 unique IDs")
    excluded = {str(value) for value in config["excluded_task_ids"]}
    dev = set(config["development_seed_trajectory_ids"])
    registry: list[dict[str, Any]] = []
    tasks: set[str] = set()
    for trajectory_id in selected:
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None:
            raise ValueError(f"retail seed trajectory not found: {trajectory_id}")
        task_id = str(trajectory["task_id"])
        annotation_id = content_id({"trajectory_id": trajectory_id}, "annitem_")
        if trajectory.get("domain") != "retail":
            raise ValueError(f"not a retail trace: {trajectory_id}")
        if task_id in excluded or trajectory_id in dev:
            raise ValueError(f"excluded retail seed: {trajectory_id}")
        if task_id in tasks:
            raise ValueError(f"duplicate retail task: {task_id}")
        if annotation_id not in merged_safe:
            raise ValueError(f"retail seed is not a merged safe label: {trajectory_id}")
        write_ids = set(trajectory.get("write_event_ids", []))
        if not write_ids:
            raise ValueError(f"retail seed has no state-changing write: {trajectory_id}")
        tasks.add(task_id)
        tools = [event["tool_call"]["name"] for event in trajectory["events"] if event.get("event_id") in write_ids and event.get("tool_call")]
        registry.append({
            "trajectory_id": trajectory_id,
            "annotation_id": annotation_id,
            "domain": "retail",
            "task_id": task_id,
            "source_agent": trajectory["source_agent"],
            "trial": trajectory["trial"],
            "event_count": len(trajectory["events"]),
            "write_event_ids": trajectory["write_event_ids"],
            "write_tools": tools,
            "review_source": "two_reviewer_merged",
            "review_label": "safe",
        })
    registry.sort(key=lambda row: (int(row["task_id"]), row["trajectory_id"]))
    for rank, row in enumerate(registry, 1):
        row["seed_rank"] = rank
    path = local_root / "augmentation_retail_seeds_private.jsonl"
    write_jsonl(path, registry)
    manifest = {
        "version": "v034-retail-seed-registry-1",
        "seed_count": len(registry),
        "unique_task_count": len(tasks),
        "task_ids": [row["task_id"] for row in registry],
        "source_agents": dict(Counter(row["source_agent"] for row in registry)),
        "review_sources": dict(Counter(row["review_source"] for row in registry)),
        "write_tools": dict(Counter(tool for row in registry for tool in row["write_tools"])),
        "registry_sha256": sha256_file(path),
        "development_seeds_excluded": sorted(dev),
        "official_gold_untouched": True,
    }
    write_json(local_root / "augmentation_retail_seed_manifest.json", manifest)
    return manifest


def select_scale_seeds(local_root: Path = DEFAULT_LOCAL_ROOT) -> list[dict[str, Any]]:
    """Load both 15-seed registries for the full run."""
    trajectories = _trajectories(local_root)
    result: list[dict[str, Any]] = []
    for domain in ("airline", "retail"):
        path = local_root / f"augmentation_{domain}_seeds_private.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 15 or len({row.get("task_id") for row in rows}) != 15:
            raise ValueError(f"{domain} registry must contain 15 unique tasks")
        for row in rows:
            trajectory = trajectories.get(row.get("trajectory_id"))
            if trajectory is None or trajectory.get("domain") != domain:
                raise ValueError(f"invalid {domain} registry trajectory: {row.get('trajectory_id')}")
            if not trajectory.get("write_event_ids"):
                raise ValueError(f"scale seed has no write: {trajectory['trajectory_id']}")
            result.append({"domain": domain, "trajectory": trajectory, "write_event_ids": trajectory["write_event_ids"]})
    return result


def select_fill_seeds(local_root: Path = DEFAULT_LOCAL_ROOT, config_path: Path | None = None) -> dict[str, Any]:
    """Freeze a small, predeclared reserve pool for filling rejected scale cases."""
    config = read_json(config_path or CONFIG_ROOT / "augmentation_fill.json")
    trajectories = _trajectories(local_root)
    merged_safe = _merged_safe(local_root)
    scale_ids = {
        row["trajectory_id"]
        for domain in ("airline", "retail")
        for row in read_jsonl(local_root / f"augmentation_{domain}_seeds_private.jsonl")
    }
    development_ids = set(config.get("excluded_development_trajectory_ids", []))
    selected = list(config["selected_seed_trajectory_ids"])
    if len(selected) != len(set(selected)):
        raise ValueError("fill seed config contains duplicate trajectory IDs")
    registry: list[dict[str, Any]] = []
    for rank, trajectory_id in enumerate(selected, 1):
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None:
            raise ValueError(f"fill seed trajectory not found: {trajectory_id}")
        if trajectory_id in scale_ids or trajectory_id in development_ids:
            raise ValueError(f"fill seed overlaps scale or development data: {trajectory_id}")
        ann_id = content_id({"trajectory_id": trajectory_id}, "annitem_")
        if ann_id not in merged_safe:
            raise ValueError(f"fill seed is not two-reviewer merged safe: {trajectory_id}")
        writes = set(trajectory.get("write_event_ids", []))
        if not writes:
            raise ValueError(f"fill seed has no state-changing write: {trajectory_id}")
        tools = [
            event["tool_call"]["name"]
            for event in trajectory["events"]
            if event.get("event_id") in writes and event.get("tool_call")
        ]
        registry.append({
            "seed_rank": rank,
            "trajectory_id": trajectory_id,
            "annotation_id": ann_id,
            "domain": trajectory["domain"],
            "task_id": str(trajectory["task_id"]),
            "source_agent": trajectory["source_agent"],
            "trial": trajectory["trial"],
            "event_count": len(trajectory["events"]),
            "write_event_ids": sorted(writes),
            "write_tools": tools,
            "review_source": "two_reviewer_merged",
            "review_label": "safe",
        })
    path = local_root / "augmentation_fill_seeds_private.jsonl"
    write_jsonl(path, registry)
    manifest = {
        "version": "v034-fill-seed-registry-1",
        "seed_count": len(registry),
        "by_domain": dict(Counter(row["domain"] for row in registry)),
        "unique_tasks_by_domain": {
            domain: len({row["task_id"] for row in registry if row["domain"] == domain})
            for domain in ("airline", "retail")
        },
        "source_agents": dict(Counter(row["source_agent"] for row in registry)),
        "registry_sha256": sha256_file(path),
        "scale_overlap": False,
        "development_overlap": False,
        "review_sources": {"two_reviewer_merged": len(registry)},
    }
    write_json(local_root / "augmentation_fill_seed_manifest.json", manifest)
    return manifest


def load_fill_seeds(local_root: Path = DEFAULT_LOCAL_ROOT) -> list[dict[str, Any]]:
    """Load the frozen reserve pool after validating its source trajectories."""
    trajectories = _trajectories(local_root)
    rows = read_jsonl(local_root / "augmentation_fill_seeds_private.jsonl")
    if not rows:
        raise ValueError("run select-fill-seeds before fill augmentation")
    result: list[dict[str, Any]] = []
    for row in rows:
        trajectory = trajectories.get(row.get("trajectory_id"))
        if trajectory is None or trajectory.get("domain") != row.get("domain"):
            raise ValueError(f"invalid fill registry trajectory: {row.get('trajectory_id')}")
        result.append({
            "domain": trajectory["domain"],
            "trajectory": trajectory,
            "write_event_ids": trajectory["write_event_ids"],
        })
    return result


def select_refill_seeds(local_root: Path = DEFAULT_LOCAL_ROOT, config_path: Path | None = None) -> dict[str, Any]:
    """Freeze a second retail-only reserve pool after human rejection of fill cases."""
    config = read_json(config_path or CONFIG_ROOT / "augmentation_refill.json")
    trajectories = _trajectories(local_root)
    merged_safe = _merged_safe(local_root)
    prior_ids = {
        row["trajectory_id"]
        for name in (
            "augmentation_airline_seeds_private.jsonl",
            "augmentation_retail_seeds_private.jsonl",
            "augmentation_fill_seeds_private.jsonl",
        )
        for row in read_jsonl(local_root / name)
    }
    development_ids = set(config.get("excluded_development_trajectory_ids", []))
    selected = list(config["selected_seed_trajectory_ids"])
    if len(selected) != len(set(selected)):
        raise ValueError("refill seed config contains duplicate trajectory IDs")
    registry: list[dict[str, Any]] = []
    tasks: set[str] = set()
    for rank, trajectory_id in enumerate(selected, 1):
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None or trajectory.get("domain") != "retail":
            raise ValueError(f"invalid retail refill seed: {trajectory_id}")
        if trajectory_id in prior_ids or trajectory_id in development_ids:
            raise ValueError(f"refill seed overlaps prior or development data: {trajectory_id}")
        task_id = str(trajectory["task_id"])
        if task_id in tasks:
            raise ValueError(f"duplicate refill task: {task_id}")
        ann_id = content_id({"trajectory_id": trajectory_id}, "annitem_")
        if ann_id not in merged_safe:
            raise ValueError(f"refill seed is not two-reviewer merged safe: {trajectory_id}")
        writes = set(trajectory.get("write_event_ids", []))
        tools = [
            event["tool_call"]["name"]
            for event in trajectory["events"]
            if event.get("event_id") in writes and event.get("tool_call")
        ]
        if not tools or "cancel_pending_order" in tools:
            raise ValueError(f"refill seed must expose a non-cancellation write: {trajectory_id}")
        tasks.add(task_id)
        registry.append({
            "seed_rank": rank,
            "trajectory_id": trajectory_id,
            "annotation_id": ann_id,
            "domain": "retail",
            "task_id": task_id,
            "source_agent": trajectory["source_agent"],
            "trial": trajectory["trial"],
            "event_count": len(trajectory["events"]),
            "write_event_ids": sorted(writes),
            "write_tools": tools,
            "review_source": "two_reviewer_merged",
            "review_label": "safe",
        })
    path = local_root / "augmentation_refill_seeds_private.jsonl"
    write_jsonl(path, registry)
    manifest = {
        "version": "v034-refill-seed-registry-1",
        "seed_count": len(registry),
        "by_domain": {"retail": len(registry)},
        "unique_task_count": len(tasks),
        "task_ids": sorted(tasks, key=int),
        "source_agents": dict(Counter(row["source_agent"] for row in registry)),
        "write_tools": dict(Counter(tool for row in registry for tool in row["write_tools"])),
        "registry_sha256": sha256_file(path),
        "prior_overlap": False,
        "development_overlap": False,
        "review_sources": {"two_reviewer_merged": len(registry)},
    }
    write_json(local_root / "augmentation_refill_seed_manifest.json", manifest)
    return manifest


def load_refill_seeds(local_root: Path = DEFAULT_LOCAL_ROOT) -> list[dict[str, Any]]:
    trajectories = _trajectories(local_root)
    rows = read_jsonl(local_root / "augmentation_refill_seeds_private.jsonl")
    if not rows:
        raise ValueError("run select-refill-seeds before refill augmentation")
    result: list[dict[str, Any]] = []
    for row in rows:
        trajectory = trajectories.get(row.get("trajectory_id"))
        if trajectory is None or trajectory.get("domain") != "retail":
            raise ValueError(f"invalid refill registry trajectory: {row.get('trajectory_id')}")
        result.append({"domain": "retail", "trajectory": trajectory, "write_event_ids": trajectory["write_event_ids"]})
    return result
