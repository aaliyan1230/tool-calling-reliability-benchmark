#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request


def _build_prompt(payload: dict) -> str:
    task = payload.get("task", {})
    available_tools = list(payload.get("available_tools", []))
    attempted_tools = list(payload.get("attempted_tools", []))
    last_status = payload.get("last_status")

    return "\n".join(
        [
            "You are a tool-selection model for an agent benchmark.",
            "Return ONLY valid JSON with this exact schema:",
            '{"tool_name":"<exact tool name>"}',
            "",
            "Rules:",
            "1) Pick exactly one tool name from AVAILABLE_TOOLS.",
            "2) Prefer tools whose schema likely matches REQUIRED_SCHEMA.",
            "3) If LAST_STATUS indicates schema issue, avoid already attempted tools.",
            "4) Do not include explanation or markdown.",
            "",
            f"TASK_ID: {task.get('task_id', '')}",
            f"POLICY: {payload.get('policy', '')}",
            f"ATTEMPT_NUMBER: {payload.get('attempt_number', '')}",
            f"PRIMARY_TOOL: {task.get('primary_tool', '')}",
            f"FALLBACK_TOOLS: {json.dumps(task.get('fallback_tools', []))}",
            f"REQUIRED_SCHEMA: {json.dumps(task.get('required_schema', []))}",
            f"AVAILABLE_TOOLS: {json.dumps(available_tools)}",
            f"ATTEMPTED_TOOLS: {json.dumps(attempted_tools)}",
            f"LAST_STATUS: {last_status}",
        ]
    )


def _query_ollama(host: str, model: str, prompt: str, timeout: float) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }
    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("response", "")).strip()


def _extract_tool_name(raw: str) -> str:
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            value = str(parsed.get("tool_name", "")).strip()
            if value:
                return value
    except json.JSONDecodeError:
        pass

    match = re.search(r'"tool_name"\s*:\s*"([^"]+)"', raw)
    if match:
        return match.group(1).strip()

    first_line = raw.splitlines()[0].strip()
    if first_line:
        return first_line.split()[0]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama-backed tool selector")
    parser.add_argument("--model", required=True, help="Ollama model tag")
    parser.add_argument(
        "--host", default="http://127.0.0.1:11434", help="Ollama host URL"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--fallback-mode",
        choices=["primary", "empty"],
        default="primary",
        help="Fallback when model output cannot be parsed",
    )
    parser.add_argument(
        "--strict-errors",
        action="store_true",
        help="Exit non-zero on Ollama request/parsing failures instead of returning empty",
    )
    args = parser.parse_args()

    payload = json.load(sys.stdin)
    prompt = _build_prompt(payload)

    query_failed = False
    try:
        raw = _query_ollama(
            host=args.host, model=args.model, prompt=prompt, timeout=args.timeout
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        query_failed = True
        if args.strict_errors:
            print(f"ollama query failed: {exc}", file=sys.stderr)
            return 2
        raw = ""

    tool_name = _extract_tool_name(raw)
    if not tool_name and query_failed and args.strict_errors:
        print("ollama returned no parsable tool output", file=sys.stderr)
        return 3

    if not tool_name and args.fallback_mode == "primary":
        task = payload.get("task", {})
        tool_name = str(task.get("primary_tool", "")).strip()

    if not tool_name and args.strict_errors:
        print("no tool_name parsed from model output", file=sys.stderr)
        return 4

    print(tool_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
