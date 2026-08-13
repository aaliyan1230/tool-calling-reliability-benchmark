from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcrb.v03.providers import load_openai_api_key, load_opencode_api_key

from .prompts import prompt_bundle
from .schema import MonitorResult, MonitorView


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


def call_monitor(
    provider: str,
    view: MonitorView,
    *,
    timeout_s: int = 120,
    max_retries: int = 4,
) -> ProviderResponse:
    bundle = prompt_bundle(view)
    if provider == "deepseek":
        return _call_deepseek(bundle, timeout_s=timeout_s, max_retries=max_retries)
    if provider == "gpt":
        return _call_gpt(bundle, timeout_s=timeout_s, max_retries=max_retries)
    raise ValueError(f"unknown provider: {provider}")


def _call_deepseek(
    bundle: dict[str, str], *, timeout_s: int, max_retries: int
) -> ProviderResponse:
    body = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": bundle["system"]},
            {"role": "user", "content": bundle["user"]},
        ],
        "reasoning_effort": "high",
        "temperature": 0,
        # Reasons must fit even when the summary view has many events.  The
        # schema is still strict, so this only prevents avoidable truncation.
        "max_tokens": 2000,
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
    output = response.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(output, str):
        raise ProviderError("DeepSeek response did not contain message content")
    usage_raw = response.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
        "cached_input_tokens": int(usage_raw.get("prompt_cache_hit_tokens", 0) or 0),
    }
    return ProviderResponse(
        provider="opencode_go",
        model_id="deepseek-v4-flash",
        output_text=output,
        raw_response=response,
        usage=usage,
        latency_ms=round((time.monotonic() - started) * 1000),
        estimated_extra_cost_usd=0.0,
    )


def _call_gpt(
    bundle: dict[str, str], *, timeout_s: int, max_retries: int
) -> ProviderResponse:
    body = {
        "model": "gpt-5.6-terra",
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
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["approve", "reject", "insufficient_evidence"],
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["verdict", "evidence_ids", "reason"],
                },
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
    output = _extract_openai_output_text(response)
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
        model_id="gpt-5.6-terra",
        output_text=output,
        raw_response=response,
        usage=usage,
        latency_ms=round((time.monotonic() - started) * 1000),
        estimated_extra_cost_usd=cost,
    )


def parse_monitor_result(text: str) -> MonitorResult:
    payload = _parse_json_object(text)
    verdict = payload.get("verdict")
    if verdict not in {"approve", "reject", "insufficient_evidence"}:
        raise ValueError(f"invalid verdict: {verdict!r}")
    evidence_ids = payload.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) for item in evidence_ids
    ):
        raise ValueError("evidence_ids must be a list of strings")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if len(reason) > 2000:
        raise ValueError("reason exceeds 2000 characters")
    return MonitorResult(
        verdict=verdict,
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        reason=reason.strip(),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("could not parse monitor JSON")


def _extract_openai_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
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
                "User-Agent": "tcrb-v031-visibility-pilot/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ProviderError("provider returned a non-object JSON response")
            return payload
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")[:2000]
            last_error = ProviderError(f"HTTP {exc.code}: {body_text}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(min(2**attempt, 8))
    raise ProviderError(f"provider request failed after retries: {last_error}")
