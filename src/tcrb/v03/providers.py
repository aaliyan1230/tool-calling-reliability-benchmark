from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import prompt_bundle
from .schema import MONITOR_RESULT_JSON_SCHEMA, MonitorResult, PromptVariant, VerifierView


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model_id: str
    output_text: str
    raw_response: dict[str, Any]
    usage: dict[str, int]
    latency_ms: int
    estimated_extra_cost_usd: float


class ProviderError(RuntimeError):
    pass


def load_opencode_api_key() -> str:
    env_key = os.environ.get("OPENCODE_API_KEY")
    if env_key:
        return env_key
    auth_path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
        key = payload["opencode-go"]["key"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ProviderError(
            "OpenCode Go key not found; set OPENCODE_API_KEY or connect OpenCode Go"
        ) from exc
    if not isinstance(key, str) or not key:
        raise ProviderError("OpenCode Go credential is empty")
    return key


def load_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ProviderError("OPENAI_API_KEY is not set")
    return key


def call_monitor(
    provider: str,
    view: VerifierView,
    prompt_variant: PromptVariant,
    *,
    timeout_s: int = 120,
    max_retries: int = 4,
) -> ProviderResponse:
    bundle = prompt_bundle(view, prompt_variant)
    if provider == "deepseek":
        return _call_deepseek(bundle, timeout_s=timeout_s, max_retries=max_retries)
    if provider == "gpt":
        return _call_gpt(bundle, timeout_s=timeout_s, max_retries=max_retries)
    raise ValueError(f"unknown provider: {provider}")


def _call_deepseek(
    bundle: dict[str, object],
    *,
    timeout_s: int,
    max_retries: int,
) -> ProviderResponse:
    model_id = "deepseek-v4-flash"
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": bundle["system"]},
            {"role": "user", "content": bundle["user"]},
        ],
        "reasoning_effort": "high",
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    started = time.monotonic()
    response = _post_json(
        "https://opencode.ai/zen/go/v1/chat/completions",
        body,
        api_key=load_opencode_api_key(),
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    latency_ms = round((time.monotonic() - started) * 1000)
    try:
        output_text = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("DeepSeek response did not contain message content") from exc
    usage_raw = response.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
        "cached_input_tokens": int(
            usage_raw.get("prompt_cache_hit_tokens", 0) or 0
        ),
    }
    return ProviderResponse(
        provider="opencode_go",
        model_id=model_id,
        output_text=str(output_text),
        raw_response=response,
        usage=usage,
        latency_ms=latency_ms,
        estimated_extra_cost_usd=0.0,
    )


def _call_gpt(
    bundle: dict[str, object],
    *,
    timeout_s: int,
    max_retries: int,
) -> ProviderResponse:
    model_id = "gpt-5.6-terra"
    body = {
        "model": model_id,
        "input": [
            {"role": "system", "content": bundle["system"]},
            {"role": "user", "content": bundle["user"]},
        ],
        "reasoning": {"effort": "high"},
        "max_output_tokens": 1200,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "monitor_result",
                "strict": True,
                "schema": MONITOR_RESULT_JSON_SCHEMA,
            }
        },
    }
    started = time.monotonic()
    response = _post_json(
        "https://api.openai.com/v1/responses",
        body,
        api_key=load_openai_api_key(),
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    latency_ms = round((time.monotonic() - started) * 1000)
    output_text = _extract_openai_output_text(response)
    usage_raw = response.get("usage") or {}
    input_details = usage_raw.get("input_tokens_details") or {}
    usage = {
        "input_tokens": int(usage_raw.get("input_tokens", 0) or 0),
        "output_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        "cached_input_tokens": int(input_details.get("cached_tokens", 0) or 0),
    }
    uncached = max(0, usage["input_tokens"] - usage["cached_input_tokens"])
    cost = (
        uncached * 2.0
        + usage["cached_input_tokens"] * 0.2
        + usage["output_tokens"] * 12.0
    ) / 1_000_000
    return ProviderResponse(
        provider="openai",
        model_id=model_id,
        output_text=output_text,
        raw_response=response,
        usage=usage,
        latency_ms=latency_ms,
        estimated_extra_cost_usd=cost,
    )


def _extract_openai_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"} and isinstance(
                content.get("text"), str
            ):
                parts.append(content["text"])
    if not parts:
        raise ProviderError("OpenAI response did not contain output text")
    return "".join(parts)


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    api_key: str,
    timeout_s: int,
    max_retries: int,
) -> dict[str, Any]:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "tcrb-v03-provenance-pilot/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = response.read().decode("utf-8")
            decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                raise ProviderError("provider returned a non-object JSON response")
            return decoded
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = ProviderError(f"HTTP {exc.code}: {error_body}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(2**attempt, 8))
    raise ProviderError(f"provider request failed after retries: {last_error}")


def parse_monitor_result(output_text: str) -> MonitorResult:
    payload = _parse_json_object(output_text)
    decision = payload.get("decision")
    if decision not in {"allow", "review", "block"}:
        raise ValueError(f"invalid decision: {decision!r}")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("confidence must be a number")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    evidence_ids = payload.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) for item in evidence_ids
    ):
        raise ValueError("evidence_ids must be a list of strings")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if len(reason) > 400:
        raise ValueError("reason exceeds 400 characters")
    return MonitorResult(
        decision=decision,
        confidence=confidence,
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        reason=reason.strip(),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    errors: list[str] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if isinstance(payload, dict):
            return payload
        errors.append("decoded JSON was not an object")
    raise ValueError("could not parse result JSON: " + "; ".join(errors[:3]))
