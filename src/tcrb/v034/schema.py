from __future__ import annotations

from typing import Any


LABELS = ("safe", "policy_breaking_write", "wrong_write_not_policy_breaking", "non_write_failure", "ambiguous")
PROFILES = ("safety_monitoring_v1", "crm_handoff_v1", "compact_crm_handoff_v1", "plain_text_crm_handoff_v1")
SUMMARY_KEYS = ("user_request", "key_facts", "actions_and_results", "state_changes", "unresolved_or_risks")
WRITE_TOOLS = {
    "airline": {"book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages", "update_reservation_flights", "update_reservation_passengers"},
    "retail": {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address", "return_delivered_order_items"},
}


def validate_summary(value: Any, event_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != set(SUMMARY_KEYS):
        return ["summary must contain exactly the required keys"]
    if not isinstance(value["user_request"], str):
        errors.append("user_request must be a string")
    for key in SUMMARY_KEYS[1:]:
        items = value[key]
        if not isinstance(items, list):
            errors.append(f"{key} must be a list")
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != {"text", "source_event_ids"}:
                errors.append(f"{key}[{index}] must contain text and source_event_ids")
                continue
            if not isinstance(item["text"], str) or not item["text"].strip():
                errors.append(f"{key}[{index}].text must be non-empty")
            ids = item["source_event_ids"]
            if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x in event_ids for x in ids):
                errors.append(f"{key}[{index}] has invalid source_event_ids")
    return errors


def validate_annotation(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("label") not in LABELS:
        errors.append("invalid label")
    if value.get("confidence") not in {"low", "medium", "high"}:
        errors.append("invalid confidence")
    for key in ("violated_rule_ids", "write_event_ids", "supporting_event_ids"):
        if not isinstance(value.get(key), list) or not all(isinstance(x, str) for x in value[key]):
            errors.append(f"{key} must be a list of strings")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        errors.append("reason must be non-empty")
    return errors
