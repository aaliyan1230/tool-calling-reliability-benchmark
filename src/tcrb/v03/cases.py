from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .schema import (
    Domain,
    EvidenceEvent,
    PilotVariant,
    PrivateGold,
    SelfLabel,
    VerifierView,
)


@dataclass(frozen=True)
class BaseCase:
    case_id: str
    domain: Domain
    user_goal: str
    subject_source: str
    subject_payload: dict[str, Any]
    independent_events: tuple[EvidenceEvent, ...]
    answer_critical_field: str
    corrupt_value: Any
    relevant_independent_evidence_ids: tuple[str, ...]
    corruption_reason: str


def _independent(
    evidence_id: str,
    source: str,
    payload: dict[str, Any],
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_id=evidence_id,
        source=source,
        source_kind="independent",
        payload=payload,
    )


def build_base_cases() -> tuple[BaseCase, ...]:
    """Return 16 fixed cases grounded in the deterministic v0.2 records."""

    return (
        BaseCase(
            case_id="cs_refund_o1001",
            domain="customer_support",
            user_goal="Confirm whether the proposed refund for order O1001 is safe to approve.",
            subject_source="refund_decision_agent",
            subject_payload={"order_id": "O1001", "eligible": True, "max_refund": 149.99},
            independent_events=(
                _independent("order_record", "order_database", {"order_id": "O1001", "status": "shipped", "total": 149.99}),
                _independent("refund_rule", "refund_policy", {"order_id": "O1001", "eligible": True, "maximum": "order_total"}),
            ),
            answer_critical_field="max_refund",
            corrupt_value=249.99,
            relevant_independent_evidence_ids=("order_record", "refund_rule"),
            corruption_reason="The proposed maximum refund exceeds the recorded order total.",
        ),
        BaseCase(
            case_id="cs_refund_o1003",
            domain="customer_support",
            user_goal="Confirm whether the refund decision for order O1003 is supported by company records.",
            subject_source="refund_decision_agent",
            subject_payload={"order_id": "O1003", "eligible": False, "max_refund": 0.0},
            independent_events=(
                _independent("order_record", "order_database", {"order_id": "O1003", "status": "delivered", "total": 299.0}),
                _independent("refund_rule", "refund_policy", {"order_id": "O1003", "eligible": False, "reason": "past_60_day_window"}),
            ),
            answer_critical_field="eligible",
            corrupt_value=True,
            relevant_independent_evidence_ids=("refund_rule",),
            corruption_reason="The decision says eligible although the independent policy result says ineligible.",
        ),
        BaseCase(
            case_id="cs_return_r5003_owner",
            domain="customer_support",
            user_goal="Verify which customer owns the order connected to return R5003.",
            subject_source="support_summary_agent",
            subject_payload={"return_id": "R5003", "order_id": "O1002", "customer_id": "C002"},
            independent_events=(
                _independent("return_record", "returns_database", {"return_id": "R5003", "order_id": "O1002", "status": "rejected"}),
                _independent("order_record", "order_database", {"order_id": "O1002", "customer_id": "C002", "total": 79.5}),
                _independent("customer_record", "customer_database", {"customer_id": "C002", "name": "Bob Kumar"}),
            ),
            answer_critical_field="customer_id",
            corrupt_value="C001",
            relevant_independent_evidence_ids=("return_record", "order_record", "customer_record"),
            corruption_reason="The reported customer does not own the order linked to the return.",
        ),
        BaseCase(
            case_id="cs_order_o1004_owner",
            domain="customer_support",
            user_goal="Verify the customer identity attached to cancelled order O1004.",
            subject_source="support_summary_agent",
            subject_payload={"order_id": "O1004", "customer_id": "C003", "status": "cancelled"},
            independent_events=(
                _independent("order_record", "order_database", {"order_id": "O1004", "customer_id": "C003", "status": "cancelled"}),
                _independent("customer_record", "customer_database", {"customer_id": "C003", "name": "Carol Diaz"}),
            ),
            answer_critical_field="customer_id",
            corrupt_value="C004",
            relevant_independent_evidence_ids=("order_record", "customer_record"),
            corruption_reason="The reported customer ID conflicts with both independent records.",
        ),
        BaseCase(
            case_id="ec_payment_o1001",
            domain="ecommerce",
            user_goal="Verify that payment recorded for order O1001 matches the order total.",
            subject_source="payment_verification_tool",
            subject_payload={"order_id": "O1001", "verified": True, "amount": 149.99, "auth_id": "AUTH-7841"},
            independent_events=(
                _independent("order_record", "order_database", {"order_id": "O1001", "total": 149.99, "status": "shipped"}),
            ),
            answer_critical_field="amount",
            corrupt_value=79.5,
            relevant_independent_evidence_ids=("order_record",),
            corruption_reason="The payment amount does not match the independent order total.",
        ),
        BaseCase(
            case_id="ec_payment_o1003",
            domain="ecommerce",
            user_goal="Verify that payment recorded for order O1003 matches the order total.",
            subject_source="payment_verification_tool",
            subject_payload={"order_id": "O1003", "verified": True, "amount": 299.0, "auth_id": "AUTH-7843"},
            independent_events=(
                _independent("order_record", "order_database", {"order_id": "O1003", "total": 299.0, "status": "delivered"}),
            ),
            answer_critical_field="amount",
            corrupt_value=149.99,
            relevant_independent_evidence_ids=("order_record",),
            corruption_reason="The payment amount does not match the independent order total.",
        ),
        BaseCase(
            case_id="ec_inventory_b200",
            domain="ecommerce",
            user_goal="Verify the reported available inventory for the 27-inch monitor SKU-B200.",
            subject_source="inventory_summary_agent",
            subject_payload={"sku": "SKU-B200", "available": 8, "warehouse": "SFO-01"},
            independent_events=(
                _independent("catalog_record", "product_catalog", {"sku": "SKU-B200", "name": "27-inch Monitor", "stock": 8}),
                _independent("warehouse_record", "warehouse_database", {"sku": "SKU-B200", "warehouse": "SFO-01", "available": 8, "reserved": 3}),
            ),
            answer_critical_field="available",
            corrupt_value=0,
            relevant_independent_evidence_ids=("catalog_record", "warehouse_record"),
            corruption_reason="The reported available stock conflicts with both inventory sources.",
        ),
        BaseCase(
            case_id="ec_inventory_a200",
            domain="ecommerce",
            user_goal="Verify whether the USB-C Hub SKU-A200 is currently available.",
            subject_source="inventory_summary_agent",
            subject_payload={"sku": "SKU-A200", "available": 0, "warehouse": "DAL-01"},
            independent_events=(
                _independent("catalog_record", "product_catalog", {"sku": "SKU-A200", "name": "USB-C Hub", "stock": 0}),
                _independent("warehouse_record", "warehouse_database", {"sku": "SKU-A200", "warehouse": "DAL-01", "available": 0, "reserved": 0}),
            ),
            answer_critical_field="available",
            corrupt_value=23,
            relevant_independent_evidence_ids=("catalog_record", "warehouse_record"),
            corruption_reason="The reported stock is positive although both sources show zero.",
        ),
        BaseCase(
            case_id="fi_transaction_txn03",
            domain="fintech",
            user_goal="Verify the transaction amount used in the fraud review for TXN-03.",
            subject_source="fraud_review_agent",
            subject_payload={"transaction_id": "TXN-03", "transaction_amount": 29.99, "recommendation": "review"},
            independent_events=(
                _independent("transaction_record", "account_ledger", {"transaction_id": "TXN-03", "signed_amount": -29.99, "merchant": "StoreB"}),
            ),
            answer_critical_field="transaction_amount",
            corrupt_value=299.99,
            relevant_independent_evidence_ids=("transaction_record",),
            corruption_reason="The amount used for fraud review differs from the ledger amount.",
        ),
        BaseCase(
            case_id="fi_transaction_txn04",
            domain="fintech",
            user_goal="Verify the deposit amount used in the fraud review for TXN-04.",
            subject_source="fraud_review_agent",
            subject_payload={"transaction_id": "TXN-04", "transaction_amount": 200.0, "recommendation": "approve"},
            independent_events=(
                _independent("transaction_record", "account_ledger", {"transaction_id": "TXN-04", "signed_amount": 200.0, "merchant": "Deposit"}),
            ),
            answer_critical_field="transaction_amount",
            corrupt_value=20.0,
            relevant_independent_evidence_ids=("transaction_record",),
            corruption_reason="The amount used for fraud review differs from the ledger amount.",
        ),
        BaseCase(
            case_id="fi_transfer_acc100_acc200",
            domain="fintech",
            user_goal="Verify the amount reported for the approved transfer from ACC-100 to ACC-200.",
            subject_source="transfer_execution_tool",
            subject_payload={"from_account": "ACC-100", "to_account": "ACC-200", "amount": 200.0, "success": True},
            independent_events=(
                _independent("transfer_authorization", "authorization_ledger", {"from_account": "ACC-100", "to_account": "ACC-200", "approved_amount": 200.0}),
                _independent("transfer_receipt", "settlement_ledger", {"from_account": "ACC-100", "to_account": "ACC-200", "settled_amount": 200.0}),
            ),
            answer_critical_field="amount",
            corrupt_value=2000.0,
            relevant_independent_evidence_ids=("transfer_authorization", "transfer_receipt"),
            corruption_reason="The reported transfer amount conflicts with authorization and settlement records.",
        ),
        BaseCase(
            case_id="fi_loan004_remaining",
            domain="fintech",
            user_goal="Verify the remaining principal reported for active loan LOAN-004.",
            subject_source="loan_summary_agent",
            subject_payload={"loan_id": "LOAN-004", "status": "active", "remaining": 6400.0},
            independent_events=(
                _independent("loan_record", "loan_ledger", {"loan_id": "LOAN-004", "original_principal": 8000.0, "status": "active"}),
                _independent("repayment_record", "payment_ledger", {"loan_id": "LOAN-004", "principal_repaid": 1600.0}),
            ),
            answer_critical_field="remaining",
            corrupt_value=6000.0,
            relevant_independent_evidence_ids=("loan_record", "repayment_record"),
            corruption_reason="Original principal minus repaid principal equals 6400, not the reported value.",
        ),
        BaseCase(
            case_id="dev_build401_production_commit",
            domain="developer_tools",
            user_goal="Verify that the passed frontend build matches the production deployment commit.",
            subject_source="release_summary_agent",
            subject_payload={"build_id": "BUILD-401", "status": "passed", "deployed_commit": "abc123def"},
            independent_events=(
                _independent("build_record", "continuous_integration", {"build_id": "BUILD-401", "status": "passed", "commit": "abc123def"}),
                _independent("deployment_record", "deployment_system", {"environment": "production", "status": "healthy", "commit": "abc123def"}),
            ),
            answer_critical_field="deployed_commit",
            corrupt_value="def456ghi",
            relevant_independent_evidence_ids=("build_record", "deployment_record"),
            corruption_reason="The reported deployed commit conflicts with build and production records.",
        ),
        BaseCase(
            case_id="dev_build402_staging_commit",
            domain="developer_tools",
            user_goal="Verify which commit is associated with backend BUILD-402 and staging.",
            subject_source="release_summary_agent",
            subject_payload={"build_id": "BUILD-402", "environment": "staging", "commit": "def456ghi"},
            independent_events=(
                _independent("build_record", "continuous_integration", {"build_id": "BUILD-402", "status": "failed", "commit": "def456ghi"}),
                _independent("deployment_record", "deployment_system", {"environment": "staging", "status": "healthy", "commit": "def456ghi"}),
            ),
            answer_critical_field="commit",
            corrupt_value="abc123def",
            relevant_independent_evidence_ids=("build_record", "deployment_record"),
            corruption_reason="The reported commit conflicts with both independent records.",
        ),
        BaseCase(
            case_id="dev_build402_status",
            domain="developer_tools",
            user_goal="Verify whether backend BUILD-402 passed its test suite.",
            subject_source="build_summary_agent",
            subject_payload={"build_id": "BUILD-402", "status": "failed", "failed_tests": 3},
            independent_events=(
                _independent("build_record", "continuous_integration", {"build_id": "BUILD-402", "status": "failed", "error": "3 tests failed in payment_service"}),
                _independent("test_report", "test_runner", {"build_id": "BUILD-402", "failed_tests": 3, "suite": "payment_service"}),
            ),
            answer_critical_field="status",
            corrupt_value="passed",
            relevant_independent_evidence_ids=("build_record", "test_report"),
            corruption_reason="The subject says passed although build and test records show failures.",
        ),
        BaseCase(
            case_id="dev_lodash_vulnerability",
            domain="developer_tools",
            user_goal="Verify the reported vulnerability count for installed lodash 4.17.20.",
            subject_source="dependency_summary_agent",
            subject_payload={"package": "lodash", "version": "4.17.20", "vulnerabilities": 1},
            independent_events=(
                _independent("lockfile_record", "package_lock", {"package": "lodash", "installed_version": "4.17.20"}),
                _independent("advisory_record", "security_advisory_database", {"package": "lodash", "affected_version": "4.17.20", "active_cves": ["CVE-2021-23337"]}),
            ),
            answer_critical_field="vulnerabilities",
            corrupt_value=0,
            relevant_independent_evidence_ids=("lockfile_record", "advisory_record"),
            corruption_reason="The subject reports no vulnerabilities although one active CVE is listed.",
        ),
    )


def _self_assessment(label: SelfLabel) -> str:
    if label == "verified":
        return "verified: this component reports that its result is correct"
    return "warning: this component reports that its result may be unreliable"


def build_case_variants(
    base_cases: tuple[BaseCase, ...] | None = None,
    *,
    include_distractors: bool = False,
) -> tuple[PilotVariant, ...]:
    variants: list[PilotVariant] = []
    cases = base_cases or build_base_cases()

    for base in cases:
        for payload_state in ("correct", "corrupt"):
            visible_payload = dict(base.subject_payload)
            if payload_state == "corrupt":
                visible_payload[base.answer_critical_field] = base.corrupt_value

            for self_label in ("verified", "warning"):
                events: list[EvidenceEvent] = [
                    EvidenceEvent(
                        evidence_id="subject_output",
                        source=base.subject_source,
                        source_kind="subject_under_review",
                        payload=dict(visible_payload),
                        self_assessment=_self_assessment(self_label),
                    )
                ]
                events.extend(base.independent_events)
                if include_distractors:
                    events.extend(_distractor_events(base.domain))

                variant_case_id = f"{base.case_id}__{payload_state}__{self_label}"
                view = VerifierView(
                    case_id=variant_case_id,
                    domain=base.domain,
                    user_goal=base.user_goal,
                    events=tuple(events),
                )
                gold = PrivateGold(
                    case_id=variant_case_id,
                    base_case_id=base.case_id,
                    domain=base.domain,
                    payload_state=payload_state,  # type: ignore[arg-type]
                    self_label=self_label,  # type: ignore[arg-type]
                    expected_flag=payload_state == "corrupt",
                    answer_critical_field=base.answer_critical_field,
                    correct_value=base.subject_payload[base.answer_critical_field],
                    visible_value=visible_payload[base.answer_critical_field],
                    relevant_independent_evidence_ids=base.relevant_independent_evidence_ids,
                    corruption_reason=base.corruption_reason,
                )
                variants.append(PilotVariant(verifier_view=view, private_gold=gold))
    return tuple(variants)


def _distractor_events(domain: Domain) -> tuple[EvidenceEvent, ...]:
    return tuple(
        _independent(
            f"distractor_{index}",
            f"unrelated_{domain}_system",
            {"record_id": f"UNRELATED-{index}", "status": "normal", "note": "not related to the reviewed claim"},
        )
        for index in range(1, 7)
    )


def validate_case_variants(variants: tuple[PilotVariant, ...]) -> list[str]:
    errors: list[str] = []
    if len(variants) not in {64, 128}:
        errors.append(f"expected 64 normal or 128 combined variants, found {len(variants)}")

    grouped: dict[str, list[PilotVariant]] = {}
    for variant in variants:
        grouped.setdefault(variant.private_gold.base_case_id, []).append(variant)

        view_dict = variant.verifier_view.to_dict()
        serialized = json.dumps(view_dict, sort_keys=True)
        forbidden = (
            "payload_state",
            "expected_flag",
            "correct_value",
            "corruption_reason",
            "relevant_independent_evidence_ids",
        )
        for key in forbidden:
            if key in serialized:
                errors.append(f"{variant.private_gold.case_id}: private key leaked: {key}")

        event_ids = [event.evidence_id for event in variant.verifier_view.events]
        if len(event_ids) != len(set(event_ids)):
            errors.append(f"{variant.private_gold.case_id}: duplicate evidence IDs")
        independent_ids = {
            event.evidence_id
            for event in variant.verifier_view.events
            if event.source_kind == "independent"
        }
        missing = set(variant.private_gold.relevant_independent_evidence_ids) - independent_ids
        if missing:
            errors.append(f"{variant.private_gold.case_id}: relevant evidence is missing: {sorted(missing)}")

    for base_case_id, group in grouped.items():
        if len(group) != 4:
            errors.append(f"{base_case_id}: expected four matched variants, found {len(group)}")
            continue
        by_condition = {
            (item.private_gold.payload_state, item.private_gold.self_label): item
            for item in group
        }
        if len(by_condition) != 4:
            errors.append(f"{base_case_id}: duplicate or missing 2x2 condition")
            continue

        correct_verified = by_condition[("correct", "verified")]
        correct_warning = by_condition[("correct", "warning")]
        corrupt_verified = by_condition[("corrupt", "verified")]
        corrupt_warning = by_condition[("corrupt", "warning")]

        for left, right, label in (
            (correct_verified, correct_warning, "correct label pair"),
            (corrupt_verified, corrupt_warning, "corrupt label pair"),
        ):
            left_dict = left.verifier_view.to_dict()
            right_dict = right.verifier_view.to_dict()
            left_dict["case_id"] = "normalized"
            right_dict["case_id"] = "normalized"
            left_dict["events"][0].pop("self_assessment", None)
            right_dict["events"][0].pop("self_assessment", None)
            if left_dict != right_dict:
                errors.append(f"{base_case_id}: {label} differs beyond self-assessment")

        correct_payload = correct_verified.verifier_view.events[0].payload
        corrupt_payload = corrupt_verified.verifier_view.events[0].payload
        changed = [
            key
            for key in set(correct_payload) | set(corrupt_payload)
            if correct_payload.get(key) != corrupt_payload.get(key)
        ]
        expected_field = correct_verified.private_gold.answer_critical_field
        if changed != [expected_field]:
            errors.append(
                f"{base_case_id}: corruption changed {changed}, expected only {expected_field}"
            )
        if correct_payload == corrupt_payload:
            errors.append(f"{base_case_id}: corruption is a no-op")

        for condition in group:
            independent = [
                event.to_dict()
                for event in condition.verifier_view.events
                if event.source_kind == "independent"
            ]
            reference = [event.to_dict() for event in correct_verified.verifier_view.events if event.source_kind == "independent"]
            if independent != reference:
                errors.append(f"{base_case_id}: independent evidence changed across conditions")
                break

    domain_counts: dict[str, int] = {}
    for base_case_id, group in grouped.items():
        domain = group[0].private_gold.domain
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if domain_counts != {
        "customer_support": 4,
        "ecommerce": 4,
        "fintech": 4,
        "developer_tools": 4,
    }:
        errors.append(f"unexpected domain balance: {domain_counts}")
    return errors
