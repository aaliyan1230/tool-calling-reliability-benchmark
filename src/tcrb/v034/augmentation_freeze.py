"""Freeze reviewed synthetic safe/unsafe pairs for the summary experiment."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .augmentation import (
    AUGMENTATION_VERSION,
    latest_result_rows,
    load_augmentation_config,
    pipeline_resource_hash,
    redundant_tool_result_audit,
    result_cache_eligible,
    seed_set_hash,
    synchronize_redundant_tool_results,
    write_event_ids,
)
from .schema import LABELS
from .util import (
    DEFAULT_LOCAL_ROOT,
    DEFAULT_RUN_ROOT,
    content_id,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


FINAL_DIR_NAME = "augmentation_final"
SOURCE_POOLS = ("scale", "fill", "refill")


def _completed_reviews(run_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    packet_paths = [run_dir / "augmentation_review_packet.jsonl"]
    supplement_packet = run_dir / "augmentation_review_packet_supplement.jsonl"
    if supplement_packet.exists():
        packet_paths.append(supplement_packet)
    packets = [row for path in packet_paths for row in read_jsonl(path)]
    reviews = [
        row
        for path in (
            run_dir / "augmentation_review_template.jsonl",
            run_dir / "augmentation_review_template_supplement.jsonl",
        )
        for row in read_jsonl(path)
    ]
    errors: list[str] = []
    if not packets:
        return {}, [f"missing review packet: {packet_paths[0]}"]
    packet_by_id = {row.get("review_id"): row for row in packets}
    review_by_id = {row.get("review_id"): row for row in reviews}
    if len(packet_by_id) != len(packets):
        errors.append(f"duplicate review ID in {run_dir}")
    if set(packet_by_id) != set(review_by_id):
        errors.append(f"completed reviews do not exactly match packets in {run_dir}")
    result: dict[str, dict[str, Any]] = {}
    for review_id, packet in packet_by_id.items():
        review = review_by_id.get(review_id) or {}
        label = review.get("label")
        if label not in LABELS:
            errors.append(f"{review_id}: incomplete or invalid label")
            continue
        if review.get("confidence") not in {"low", "medium", "high"}:
            errors.append(f"{review_id}: invalid confidence")
        if not isinstance(review.get("reason"), str) or not review["reason"].strip():
            errors.append(f"{review_id}: missing reason")
        trace = packet.get("augmented_trace") or {}
        event_ids = {event.get("event_id") for event in trace.get("events", [])}
        write_ids = set(write_event_ids(trace))
        rule_ids = {rule.get("id") for rule in packet.get("policy_rules", [])}
        for key in ("violated_rule_ids", "write_event_ids", "supporting_event_ids"):
            values = review.get(key)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                errors.append(f"{review_id}: {key} must be a string list")
        if not set(review.get("violated_rule_ids") or []).issubset(rule_ids):
            errors.append(f"{review_id}: unknown violated policy rule")
        if not set(review.get("write_event_ids") or []).issubset(write_ids):
            errors.append(f"{review_id}: write evidence is not a state-changing call")
        if not set(review.get("supporting_event_ids") or []).issubset(event_ids):
            errors.append(f"{review_id}: supporting evidence is not in the augmented trace")
        if label == "policy_breaking_write" and not (
            review.get("violated_rule_ids") and review.get("write_event_ids") and review.get("supporting_event_ids")
        ):
            errors.append(f"{review_id}: policy-breaking label needs rule, write, and supporting evidence")
        original_id = (packet.get("original_trace") or {}).get("trajectory_id")
        if original_id:
            result[original_id] = {"packet": packet, "review": review}
    return result, errors


def _pool_candidates(local_root: Path, run_root: Path, pool: str) -> tuple[list[dict[str, Any]], list[str]]:
    run_dir = run_root / f"augmentation_{pool}"
    try:
        config_hash = seed_set_hash(pool, local_root)
    except Exception as exc:
        return [], [f"{pool}: {type(exc).__name__}: {exc}"]
    latest = latest_result_rows(run_dir / "pilot_results.jsonl", config_hash)
    reviews, errors = _completed_reviews(run_dir)
    candidates: list[dict[str, Any]] = []
    config = load_augmentation_config()
    resource_hash = pipeline_resource_hash()
    for source_id, bundle in reviews.items():
        result = latest.get(source_id)
        review = bundle["review"]
        if review.get("label") != "policy_breaking_write":
            continue
        if not result_cache_eligible(result, resource_hash, config):
            errors.append(f"{pool}:{source_id}: reviewed mutation does not pass current technical gates")
            continue
        original = result["trajectory"]
        augmented = result["augmented_trajectory"]
        if original.get("trajectory_id") != source_id:
            errors.append(f"{pool}:{source_id}: source trajectory mismatch")
            continue
        candidates.append({
            "pool": pool,
            "source_trajectory_id": source_id,
            "augmented_trajectory_id": augmented["trajectory_id"],
            "domain": result["domain"],
            "task_id": str(bundle["packet"].get("original_trace", {}).get("task_id") or ""),
            "result": result,
            "review": review,
        })
    return candidates, errors


def _source_metadata(local_root: Path) -> dict[str, dict[str, Any]]:
    return {
        row["trajectory_id"]: row
        for row in read_jsonl(local_root / "normalized" / "trajectories.jsonl")
    }


def _select(candidates: list[dict[str, Any]], source: dict[str, dict[str, Any]], per_domain: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for domain in ("airline", "retail"):
        domain_rows = [row for row in candidates if row["domain"] == domain]
        for row in domain_rows:
            metadata = source[row["source_trajectory_id"]]
            row["task_id"] = str(metadata["task_id"])
            row["source_agent"] = metadata["source_agent"]
        chosen: list[dict[str, Any]] = []
        used_tasks: set[str] = set()
        counts: Counter[str] = Counter()
        while len(chosen) < per_domain:
            available = [row for row in domain_rows if row not in chosen and row["task_id"] not in used_tasks]
            if not available:
                raise ValueError(
                    f"only {len(chosen)} unique reviewed {domain} synthetic pairs are available; complete more fill reviews"
                )
            # Keep validated scale cases first. Fill cases replace only scale failures.
            pick = min(
                available,
                key=lambda row: (
                    row["pool"] != "scale",
                    counts[row["source_agent"]],
                    content_id({"source": row["source_trajectory_id"], "unsafe": row["augmented_trajectory_id"]}),
                ),
            )
            chosen.append(pick)
            used_tasks.add(pick["task_id"])
            counts[pick["source_agent"]] += 1
        selected.extend(chosen)
    return selected


def _summary_trajectory(trace: dict[str, Any], metadata: dict[str, Any], *, synthetic: bool) -> dict[str, Any]:
    events = trace["events"]
    synchronized_ids: list[str] = []
    if synthetic:
        events, synchronized_ids = synchronize_redundant_tool_results(metadata["events"], events)
    value = {
        "trajectory_id": trace["trajectory_id"],
        "domain": metadata["domain"],
        "task_id": str(metadata["task_id"]),
        "source_agent": metadata["source_agent"],
        "trial": metadata["trial"],
        "events": events,
    }
    value["write_event_ids"] = write_event_ids(value)
    value["provenance"] = "synthetic_mutation" if synthetic else "natural_safe_source"
    value["synchronized_redundant_tool_event_ids"] = synchronized_ids
    return value


def _development_pairs(local_root: Path, run_root: Path, source: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the four excluded augmentation-pilot pairs only for prompt smoke tests."""
    config_hash = seed_set_hash("pilot", local_root)
    latest = latest_result_rows(run_root / "augmentation_pilot" / "pilot_results.jsonl", config_hash)
    config = load_augmentation_config()
    resource_hash = pipeline_resource_hash()
    records = [
        row for row in latest.values()
        if result_cache_eligible(row, resource_hash, config)
    ]
    if len(records) != 4 or Counter(row.get("domain") for row in records) != Counter({"airline": 2, "retail": 2}):
        raise ValueError("the excluded four-pair augmentation pilot is not technically ready for summary smoke testing")
    trajectories: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for row in records:
        metadata = source[row["trajectory_id"]]
        safe = _summary_trajectory(row["trajectory"], metadata, synthetic=False)
        unsafe = _summary_trajectory(row["augmented_trajectory"], metadata, synthetic=True)
        pair_id = content_id({"safe": safe["trajectory_id"], "unsafe": unsafe["trajectory_id"], "development": True}, "augdev_")
        trajectories.extend((safe, unsafe))
        pairs.append({
            "pair_id": pair_id,
            "domain": metadata["domain"],
            "task_id": str(metadata["task_id"]),
            "source_agent": metadata["source_agent"],
            "safe_candidate_id": safe["trajectory_id"],
            "unsafe_candidate_id": unsafe["trajectory_id"],
            "augmentation_pool": "pilot_development",
        })
    return sorted(trajectories, key=lambda row: row["trajectory_id"]), sorted(pairs, key=lambda row: row["pair_id"])


def freeze_augmented_dataset(
    local_root: Path = DEFAULT_LOCAL_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
    pairs_per_domain: int = 15,
) -> dict[str, Any]:
    """Freeze exactly matched, reviewed safe/synthetic-unsafe pairs."""
    if pairs_per_domain <= 0:
        raise ValueError("pairs_per_domain must be positive")
    candidates: list[dict[str, Any]] = []
    review_errors: list[str] = []
    for pool in SOURCE_POOLS:
        rows, errors = _pool_candidates(local_root, run_root, pool)
        candidates.extend(rows)
        review_errors.extend(errors)
    if review_errors:
        raise ValueError("augmentation review is not freeze-ready:\n- " + "\n- ".join(review_errors))
    source = _source_metadata(local_root)
    selected = _select(candidates, source, pairs_per_domain)
    development_trajectories, development_pairs = _development_pairs(local_root, run_root, source)
    seed_registry = {
        row["trajectory_id"]: row
        for domain in ("airline", "retail")
        for row in read_jsonl(local_root / f"augmentation_{domain}_seeds_private.jsonl")
        + read_jsonl(local_root / "augmentation_fill_seeds_private.jsonl")
        + read_jsonl(local_root / "augmentation_refill_seeds_private.jsonl")
    }
    for row in selected:
        seed = seed_registry.get(row["source_trajectory_id"])
        if not seed or seed.get("review_label") != "safe":
            raise ValueError(f"source trace is not registered as reviewed safe: {row['source_trajectory_id']}")
    trajectories: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for row in selected:
        metadata = source[row["source_trajectory_id"]]
        safe = _summary_trajectory(row["result"]["trajectory"], metadata, synthetic=False)
        unsafe = _summary_trajectory(row["result"]["augmented_trajectory"], metadata, synthetic=True)
        pair_id = content_id({"safe": safe["trajectory_id"], "unsafe": unsafe["trajectory_id"]}, "augpair_")
        pairs.append({
            "pair_id": pair_id,
            "domain": row["domain"],
            "task_id": row["task_id"],
            "source_agent": row["source_agent"],
            "safe_candidate_id": safe["trajectory_id"],
            "unsafe_candidate_id": unsafe["trajectory_id"],
            "augmentation_pool": row["pool"],
        })
        gold.append({
            "pair_id": pair_id,
            "safe_trajectory_id": safe["trajectory_id"],
            "unsafe_trajectory_id": unsafe["trajectory_id"],
            "safe_label": "safe",
            "unsafe_label": "policy_breaking_write",
            "augmentation_pool": row["pool"],
            "augmentation_version": AUGMENTATION_VERSION,
            "human_review": row["review"],
            "source_safe_review": {
                "review_source": seed_registry[safe["trajectory_id"]].get("review_source"),
            },
        })
        trajectories.extend((safe, unsafe))
    trajectories.sort(key=lambda row: row["trajectory_id"])
    pairs.sort(key=lambda row: (row["domain"], int(row["task_id"]), row["pair_id"]))
    gold.sort(key=lambda row: row["pair_id"])
    final_dir = local_root / FINAL_DIR_NAME
    trajectories_path = final_dir / "trajectories.jsonl"
    pairs_path = final_dir / "frozen_pairs_private.jsonl"
    public_path = final_dir / "frozen_pairs_public.jsonl"
    gold_path = final_dir / "private_gold.jsonl"
    development_trajectories_path = final_dir / "development_trajectories.jsonl"
    development_pairs_path = final_dir / "dev_pairs_private.jsonl"
    write_jsonl(trajectories_path, trajectories)
    write_jsonl(pairs_path, pairs)
    write_jsonl(public_path, [
        {"pair_id": row["pair_id"], "domain": row["domain"], "task_id": row["task_id"], "trajectory_ids": sorted([row["safe_candidate_id"], row["unsafe_candidate_id"]])}
        for row in pairs
    ])
    write_jsonl(gold_path, gold)
    write_jsonl(development_trajectories_path, development_trajectories)
    write_jsonl(development_pairs_path, development_pairs)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in (
            trajectories_path,
            pairs_path,
            public_path,
            gold_path,
            development_trajectories_path,
            development_pairs_path,
        )
    }
    manifest = {
        "version": "v034-augmented-frozen-1",
        "gold_frozen": True,
        "final_pairs": len(pairs),
        "trajectory_count": len(trajectories),
        "development_pairs": len(development_pairs),
        "by_domain": dict(Counter(row["domain"] for row in pairs)),
        "by_pool": dict(Counter(row["augmentation_pool"] for row in pairs)),
        "source_agents": dict(Counter(row["source_agent"] for row in pairs)),
        "unique_tasks_by_domain": {
            domain: len({row["task_id"] for row in pairs if row["domain"] == domain})
            for domain in ("airline", "retail")
        },
        "natural_and_synthetic_kept_separate": True,
        "selection_used_summary_or_monitor_results": False,
        "artifact_sha256": artifact_hashes,
    }
    write_json(final_dir / "manifest.json", manifest)
    return manifest


def audit_augmented_dataset(local_root: Path = DEFAULT_LOCAL_ROOT) -> dict[str, Any]:
    final_dir = local_root / FINAL_DIR_NAME
    errors: list[str] = []
    pairs = read_jsonl(final_dir / "frozen_pairs_private.jsonl")
    trajectories = read_jsonl(final_dir / "trajectories.jsonl")
    gold = read_jsonl(final_dir / "private_gold.jsonl")
    development_pairs = read_jsonl(final_dir / "dev_pairs_private.jsonl")
    development_trajectories = read_jsonl(final_dir / "development_trajectories.jsonl")
    manifest_path = final_dir / "manifest.json"
    if not manifest_path.exists():
        return {"version": "v034-augmented-audit-1", "passed": False, "errors": ["augmented dataset is not frozen"]}
    from .util import read_json
    manifest = read_json(manifest_path)
    by_id = {row.get("trajectory_id"): row for row in trajectories}
    if len(by_id) != len(trajectories):
        errors.append("duplicate trajectory IDs")
    if len(pairs) != 30 or Counter(row.get("domain") for row in pairs) != Counter({"airline": 15, "retail": 15}):
        errors.append("dataset is not exactly 15 airline + 15 retail pairs")
    for pair in pairs:
        safe = by_id.get(pair.get("safe_candidate_id"))
        unsafe = by_id.get(pair.get("unsafe_candidate_id"))
        if not safe or not unsafe:
            errors.append(f"{pair.get('pair_id')}: missing trajectory")
            continue
        if safe.get("provenance") != "natural_safe_source" or unsafe.get("provenance") != "synthetic_mutation":
            errors.append(f"{pair.get('pair_id')}: provenance mismatch")
        if (safe.get("domain"), safe.get("task_id")) != (unsafe.get("domain"), unsafe.get("task_id")):
            errors.append(f"{pair.get('pair_id')}: pair domain/task mismatch")
        for trajectory in (safe, unsafe):
            event_ids = [event.get("event_id") for event in trajectory.get("events", [])]
            if len(event_ids) != len(set(event_ids)):
                errors.append(f"{trajectory.get('trajectory_id')}: duplicate event IDs")
        duplicate_audit = redundant_tool_result_audit(safe["events"], unsafe["events"])
        if not duplicate_audit["passed"]:
            errors.append(f"{pair.get('pair_id')}: conflicting duplicate tool results")
    if len(gold) != len(pairs):
        errors.append("private gold count does not match pair count")
    if len(development_pairs) != 4 or Counter(row.get("domain") for row in development_pairs) != Counter({"airline": 2, "retail": 2}):
        errors.append("development set is not exactly two pairs per domain")
    final_ids = set(by_id)
    development_ids = {row.get("trajectory_id") for row in development_trajectories}
    if final_ids & development_ids:
        errors.append("development and final trajectory IDs overlap")
    final_tasks = {(row.get("domain"), row.get("task_id")) for row in trajectories}
    development_tasks = {(row.get("domain"), row.get("task_id")) for row in development_trajectories}
    if final_tasks & development_tasks:
        errors.append("development and final tasks overlap")
    from .audit import nested_keys
    from .summaries import FORBIDDEN_INPUT_KEYS, action_receipt, source_input
    for trajectory in trajectories + development_trajectories:
        leaked = nested_keys(source_input(trajectory)) & FORBIDDEN_INPUT_KEYS
        if leaked:
            errors.append(f"{trajectory.get('trajectory_id')}: private summary-input fields: {sorted(leaked)}")
        receipt = action_receipt(trajectory)
        for write in receipt["writes"]:
            if not write.get("result_event_id") or write.get("result") is None:
                errors.append(f"{trajectory.get('trajectory_id')}: write receipt is missing a linked result")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        path = final_dir / name
        if not path.exists() or sha256_file(path) != expected:
            errors.append(f"artifact hash mismatch: {name}")
    return {
        "version": "v034-augmented-audit-1",
        "passed": not errors,
        "errors": errors,
        "pairs": len(pairs),
        "trajectories": len(trajectories),
    }
