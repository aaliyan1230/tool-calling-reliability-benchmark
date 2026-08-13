"""Versioned registries for policies and monitor model settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from tcrb.v034.util import CONFIG_ROOT, read_json, sha256_bytes


POLICY_CONFIG = CONFIG_ROOT.parent / "v035" / "policy_bundles.json"
MODEL_CONFIG = CONFIG_ROOT.parent / "v035" / "monitor_models.json"


@lru_cache(maxsize=1)
def policy_registry() -> dict[str, Any]:
    return read_json(POLICY_CONFIG)


@lru_cache(maxsize=1)
def model_registry() -> dict[str, Any]:
    return read_json(MODEL_CONFIG)


def policy_bundle(mode: str) -> list[dict[str, str]]:
    registry = policy_registry()
    try:
        rules = registry["bundles"][mode]["rules"]
    except KeyError as exc:
        raise ValueError(f"unknown policy mode: {mode}") from exc
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"policy bundle {mode!r} is empty")
    normalized = [{"id": str(rule["id"]), "text": str(rule["text"])} for rule in rules]
    ids = [rule["id"] for rule in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"policy bundle {mode!r} contains duplicate IDs")
    return normalized


def policy_bundle_version(mode: str) -> str:
    registry = policy_registry()
    if mode not in registry["bundles"]:
        raise ValueError(f"unknown policy mode: {mode}")
    return f"{registry['version']}:{mode}"


def policy_bundle_sha256(mode: str) -> str:
    import json

    value = {"version": policy_bundle_version(mode), "rules": policy_bundle(mode)}
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def model_settings(model: str) -> dict[str, Any]:
    try:
        settings = model_registry()["models"][model]
    except KeyError as exc:
        known = ", ".join(sorted(model_registry().get("models", {})))
        raise ValueError(f"model {model!r} is not registered; add it to {MODEL_CONFIG} (known: {known})") from exc
    required = {"endpoint", "api_key_env", "max_tokens_field", "reasoning_effort", "temperature", "max_output_tokens"}
    missing = required - set(settings)
    if missing:
        raise ValueError(f"model {model!r} is missing registry fields: {sorted(missing)}")
    normalized = dict(settings)
    normalized.setdefault("protocol", "openai_chat")
    normalized.setdefault("auth_header", "Authorization")
    if normalized["protocol"] not in {"openai_chat", "anthropic_messages"}:
        raise ValueError(f"model {model!r} has unsupported protocol: {normalized['protocol']!r}")
    if normalized["auth_header"] not in {"Authorization", "x-api-key"}:
        raise ValueError(f"model {model!r} has unsupported auth header: {normalized['auth_header']!r}")
    if normalized["protocol"] == "anthropic_messages" and normalized["max_tokens_field"] != "max_tokens":
        raise ValueError("Anthropic Messages models must use max_tokens")
    return normalized


def model_registry_sha256() -> str:
    return sha256_bytes(MODEL_CONFIG.read_bytes())
