from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_DIR = ROOT / "workloads" / "enriched"
EVAL_DIR = ROOT / "workloads" / "eval_cases"


def _tool(
    name: str,
    description: str,
    base_latency_ms: int,
    jitter_ms: int,
    timeout_ms: int,
    schema_fields: list[str],
    fault_multipliers: dict[str, float],
) -> dict:
    return {
        "name": name,
        "description": description,
        "base_latency_ms": base_latency_ms,
        "jitter_ms": jitter_ms,
        "timeout_ms": timeout_ms,
        "schema_fields": schema_fields,
        "fault_multipliers": fault_multipliers,
    }


def _task(
    task_id: str,
    user_query: str,
    primary_tool: str,
    fallback_tools: list[str],
    required_schema: list[str],
) -> dict:
    return {
        "task_id": task_id,
        "user_query": user_query,
        "primary_tool": primary_tool,
        "fallback_tools": fallback_tools,
        "required_schema": required_schema,
    }


def _build_customer_support_suite() -> tuple[dict, dict]:
    toolset_id = "customer_support"
    tools = [
        _tool("ticket_lookup", "Retrieve ticket details by id, user email, or order id.", 260, 80, 900, ["ticket_id", "status", "priority", "summary", "owner"], {"rate_limit": 1.2}),
        _tool("ticket_search", "Search historical tickets by text, issue type, and tags.", 330, 90, 1000, ["results", "query", "count", "confidence"], {"network_failure": 1.1}),
        _tool("sla_policy_resolver", "Compute SLA targets based on plan tier and incident severity.", 190, 60, 700, ["response_sla_minutes", "resolution_sla_minutes", "policy_id"], {"contract_drift": 1.1}),
        _tool("refund_policy_checker", "Validate whether an order is eligible for refund and rules.", 210, 70, 800, ["eligible", "reason", "policy_version"], {"malformed_schema": 1.2}),
        _tool("order_status_api", "Fetch shipment and fulfillment status for order ids.", 250, 70, 900, ["order_id", "shipment_status", "eta"], {"timeout": 1.1}),
        _tool("shipment_trace_api", "Get carrier events and package tracking timeline.", 370, 100, 1100, ["tracking_id", "events", "latest_status"], {"network_failure": 1.4}),
        _tool("kb_semantic_search", "Search help center knowledge base using semantic retrieval.", 420, 130, 1200, ["answer", "sources", "confidence"], {"timeout": 1.2}),
        _tool("kb_keyword_search", "Search help center articles with keyword index.", 290, 90, 900, ["answer", "sources", "confidence"], {"timeout": 0.8}),
        _tool("account_entitlement_check", "Check subscription tier and feature entitlements.", 180, 55, 700, ["account_id", "plan", "entitlements"], {"rate_limit": 1.1}),
        _tool("identity_verification", "Run KBA and OTP verification for account recovery.", 460, 120, 1400, ["verified", "method", "risk_score"], {"timeout": 1.3}),
        _tool("auth_log_reader", "Inspect login and session events for suspicious activity.", 320, 95, 1000, ["events", "risk_flags", "user_id"], {"malformed_schema": 1.1}),
        _tool("incident_status_feed", "Get current platform incidents and affected services.", 200, 60, 750, ["incident_id", "severity", "status", "components"], {"network_failure": 1.2}),
        _tool("email_drafter", "Generate customer-safe response email from context.", 340, 110, 1100, ["subject", "body", "tone"], {"malformed_schema": 1.3}),
        _tool("chat_reply_drafter", "Generate short in-chat response optimized for brevity.", 260, 90, 900, ["message", "tone", "next_steps"], {"malformed_schema": 1.2}),
        _tool("priority_router", "Assign queue and owner team based on issue class.", 170, 50, 650, ["queue", "team", "priority"], {"contract_drift": 1.1}),
        _tool("customer_sentiment_model", "Estimate sentiment and urgency from customer text.", 230, 80, 850, ["sentiment", "urgency", "confidence"], {"timeout": 1.1}),
    ]

    tasks = [
        _task("cs-001", "Customer asks where order 88271 package is right now.", "order_status_api", ["shipment_trace_api", "ticket_lookup"], ["order_id", "shipment_status", "eta"]),
        _task("cs-002", "Customer says tracking is stale and wants live carrier events.", "shipment_trace_api", ["order_status_api", "ticket_lookup"], ["tracking_id", "events", "latest_status"]),
        _task("cs-003", "Customer requests refund for delayed package under premium plan.", "refund_policy_checker", ["order_status_api", "account_entitlement_check"], ["eligible", "reason", "policy_version"]),
        _task("cs-004", "Customer asks if their plan includes priority phone support.", "account_entitlement_check", ["sla_policy_resolver", "ticket_lookup"], ["account_id", "plan", "entitlements"]),
        _task("cs-005", "Agent needs SLA response and resolution windows for sev2 incident.", "sla_policy_resolver", ["account_entitlement_check", "priority_router"], ["response_sla_minutes", "resolution_sla_minutes", "policy_id"]),
        _task("cs-006", "Customer cannot login and asks if account was breached.", "auth_log_reader", ["identity_verification", "incident_status_feed"], ["events", "risk_flags", "user_id"]),
        _task("cs-007", "Customer forgot password and needs secure identity verification.", "identity_verification", ["auth_log_reader", "ticket_lookup"], ["verified", "method", "risk_score"]),
        _task("cs-008", "Customer asks if current outage explains checkout errors.", "incident_status_feed", ["ticket_search", "kb_semantic_search"], ["incident_id", "severity", "status", "components"]),
        _task("cs-009", "Agent wants best help-center answer for invoice mismatch question.", "kb_semantic_search", ["kb_keyword_search", "ticket_search"], ["answer", "sources", "confidence"]),
        _task("cs-010", "Agent needs exact policy doc for return window keyword search.", "kb_keyword_search", ["kb_semantic_search", "ticket_search"], ["answer", "sources", "confidence"]),
        _task("cs-011", "Find similar old tickets about duplicate billing charge.", "ticket_search", ["ticket_lookup", "kb_semantic_search"], ["results", "query", "count", "confidence"]),
        _task("cs-012", "Open ticket details for ticket TCK-29011.", "ticket_lookup", ["ticket_search", "priority_router"], ["ticket_id", "status", "priority", "summary", "owner"]),
        _task("cs-013", "Route this angry enterprise customer issue to correct queue.", "priority_router", ["customer_sentiment_model", "ticket_lookup"], ["queue", "team", "priority"]),
        _task("cs-014", "Assess urgency from customer message that says app is unusable.", "customer_sentiment_model", ["priority_router", "chat_reply_drafter"], ["sentiment", "urgency", "confidence"]),
        _task("cs-015", "Draft concise in-chat response confirming we are investigating.", "chat_reply_drafter", ["email_drafter", "incident_status_feed"], ["message", "tone", "next_steps"]),
        _task("cs-016", "Draft complete follow-up email with apology and next actions.", "email_drafter", ["chat_reply_drafter", "ticket_lookup"], ["subject", "body", "tone"]),
        _task("cs-017", "Customer asks for owner and priority of existing escalation ticket.", "ticket_lookup", ["priority_router", "ticket_search"], ["ticket_id", "status", "priority", "summary", "owner"]),
        _task("cs-018", "Agent needs KB answer plus references before replying to customer.", "kb_semantic_search", ["kb_keyword_search", "email_drafter"], ["answer", "sources", "confidence"]),
    ]

    eval_cases = {
        "toolset_id": toolset_id,
        "cases": [
            {
                "case_id": f"{toolset_id}-{task['task_id']}",
                "task_id": task["task_id"],
                "question": task["user_query"],
                "expected_first_tool": task["primary_tool"],
                "expected_tool_sequence": [task["primary_tool"]],
                "acceptable_alternatives": task["fallback_tools"],
            }
            for task in tasks
        ],
    }
    workload = {"toolset_id": toolset_id, "tools": tools, "tasks": tasks}
    return workload, eval_cases


def _build_ecommerce_ops_suite() -> tuple[dict, dict]:
    toolset_id = "ecommerce_ops"
    tools = [
        _tool("catalog_search", "Search product catalog by text, attributes, and sku.", 300, 100, 1000, ["products", "query", "count", "confidence"], {"rate_limit": 1.1}),
        _tool("inventory_realtime", "Fetch real-time inventory at warehouse and store level.", 230, 75, 850, ["sku", "available", "location", "reserved"], {"network_failure": 1.2}),
        _tool("inventory_forecast", "Predict restock date and stockout risk.", 410, 120, 1200, ["sku", "restock_eta", "stockout_risk"], {"timeout": 1.2}),
        _tool("pricing_engine", "Compute dynamic selling price from strategy rules.", 270, 80, 900, ["sku", "price", "currency", "reason"], {"contract_drift": 1.1}),
        _tool("promotion_eligibility", "Determine if order qualifies for campaign discounts.", 240, 80, 850, ["eligible", "campaign_id", "reason"], {"malformed_schema": 1.2}),
        _tool("recommendation_api", "Generate personalized product recommendations.", 380, 130, 1150, ["user_id", "recommendations", "confidence"], {"timeout": 1.2}),
        _tool("shipment_quote", "Estimate shipping rates and delivery windows.", 310, 95, 1000, ["carrier", "cost", "eta", "service_level"], {"network_failure": 1.3}),
        _tool("route_optimizer", "Optimize fulfillment route and split shipments.", 470, 140, 1300, ["route", "cost", "eta", "split_plan"], {"timeout": 1.3}),
        _tool("payment_auth", "Authorize payment and return authorization token.", 260, 85, 900, ["approved", "auth_id", "risk_score"], {"rate_limit": 1.2}),
        _tool("fraud_signal_api", "Return fraud score and reasons for an order.", 350, 110, 1050, ["score", "reasons", "decision"], {"malformed_schema": 1.2}),
        _tool("return_authorization", "Issue return authorization and shipping label.", 290, 90, 950, ["rma_id", "label_url", "expires_at"], {"contract_drift": 1.1}),
        _tool("vendor_eta_feed", "Pull supplier ETA and backlog for sku replenishment.", 430, 120, 1250, ["sku", "supplier_eta", "backlog_units"], {"network_failure": 1.4}),
        _tool("tax_calculator", "Compute tax jurisdiction and total tax amount.", 220, 70, 800, ["tax_amount", "jurisdiction", "rate"], {"contract_drift": 1.1}),
        _tool("currency_fx", "Convert amount using latest market FX rates.", 190, 60, 700, ["base", "quote", "rate", "converted_amount"], {"rate_limit": 1.1}),
        _tool("seller_reputation", "Score seller reliability from fulfillment history.", 280, 90, 900, ["seller_id", "reputation", "flags"], {"malformed_schema": 1.1}),
        _tool("ops_alert_drafter", "Draft operations incident update for stakeholders.", 320, 100, 1000, ["subject", "body", "severity"], {"malformed_schema": 1.3}),
    ]

    tasks = [
        _task("eco-001", "Is sku A11 in stock right now in Dallas warehouse?", "inventory_realtime", ["inventory_forecast", "catalog_search"], ["sku", "available", "location", "reserved"]),
        _task("eco-002", "When will sku B92 be restocked and risk of stockout?", "inventory_forecast", ["vendor_eta_feed", "inventory_realtime"], ["sku", "restock_eta", "stockout_risk"]),
        _task("eco-003", "Search running shoes under 120 dollars with high rating.", "catalog_search", ["recommendation_api", "pricing_engine"], ["products", "query", "count", "confidence"]),
        _task("eco-004", "Compute dynamic price for sku C12 flash sale now.", "pricing_engine", ["promotion_eligibility", "tax_calculator"], ["sku", "price", "currency", "reason"]),
        _task("eco-005", "Check if cart qualifies for summer campaign discount.", "promotion_eligibility", ["pricing_engine", "tax_calculator"], ["eligible", "campaign_id", "reason"]),
        _task("eco-006", "Personalized cross-sell suggestions for user 8801.", "recommendation_api", ["catalog_search", "seller_reputation"], ["user_id", "recommendations", "confidence"]),
        _task("eco-007", "Quote shipping cost and ETA for order to Boston.", "shipment_quote", ["route_optimizer", "inventory_realtime"], ["carrier", "cost", "eta", "service_level"]),
        _task("eco-008", "Optimize split shipment plan for multi-warehouse order.", "route_optimizer", ["shipment_quote", "inventory_realtime"], ["route", "cost", "eta", "split_plan"]),
        _task("eco-009", "Authorize payment for order 55019 using stored card.", "payment_auth", ["fraud_signal_api", "tax_calculator"], ["approved", "auth_id", "risk_score"]),
        _task("eco-010", "Evaluate fraud risk on order 55019 before capture.", "fraud_signal_api", ["payment_auth", "seller_reputation"], ["score", "reasons", "decision"]),
        _task("eco-011", "Create return label for damaged blender item.", "return_authorization", ["promotion_eligibility", "shipment_quote"], ["rma_id", "label_url", "expires_at"]),
        _task("eco-012", "What supplier ETA do we have for sku B92 backlog?", "vendor_eta_feed", ["inventory_forecast", "inventory_realtime"], ["sku", "supplier_eta", "backlog_units"]),
        _task("eco-013", "Calculate tax for CA order subtotal 89 dollars.", "tax_calculator", ["currency_fx", "pricing_engine"], ["tax_amount", "jurisdiction", "rate"]),
        _task("eco-014", "Convert 149 EUR to USD for checkout display.", "currency_fx", ["tax_calculator", "pricing_engine"], ["base", "quote", "rate", "converted_amount"]),
        _task("eco-015", "Check if seller S-77 has trust or risk flags.", "seller_reputation", ["fraud_signal_api", "catalog_search"], ["seller_id", "reputation", "flags"]),
        _task("eco-016", "Draft severe ops incident update for delayed shipments.", "ops_alert_drafter", ["shipment_quote", "route_optimizer"], ["subject", "body", "severity"]),
        _task("eco-017", "Find low-stock products likely to stock out this week.", "inventory_forecast", ["inventory_realtime", "vendor_eta_feed"], ["sku", "restock_eta", "stockout_risk"]),
        _task("eco-018", "Can we approve payment and still run fraud checks quickly?", "fraud_signal_api", ["payment_auth", "seller_reputation"], ["score", "reasons", "decision"]),
    ]

    eval_cases = {
        "toolset_id": toolset_id,
        "cases": [
            {
                "case_id": f"{toolset_id}-{task['task_id']}",
                "task_id": task["task_id"],
                "question": task["user_query"],
                "expected_first_tool": task["primary_tool"],
                "expected_tool_sequence": [task["primary_tool"]],
                "acceptable_alternatives": task["fallback_tools"],
            }
            for task in tasks
        ],
    }
    workload = {"toolset_id": toolset_id, "tools": tools, "tasks": tasks}
    return workload, eval_cases


def _build_fintech_risk_suite() -> tuple[dict, dict]:
    toolset_id = "fintech_risk"
    tools = [
        _tool("kyc_profile_lookup", "Fetch customer KYC profile and verification status.", 260, 80, 900, ["customer_id", "kyc_status", "risk_tier"], {"network_failure": 1.1}),
        _tool("aml_screening", "Screen customer/entity against sanctions and watchlists.", 390, 120, 1200, ["match", "score", "watchlist", "reason"], {"timeout": 1.2}),
        _tool("transaction_graph", "Build graph of counterparties and money flow paths.", 520, 160, 1450, ["nodes", "edges", "anomaly_score"], {"timeout": 1.3}),
        _tool("device_fingerprint", "Assess device trust from hardware and browser signals.", 240, 80, 850, ["device_id", "trust_score", "flags"], {"malformed_schema": 1.2}),
        _tool("ip_geo_risk", "Compute geolocation and VPN/proxy risk from IP.", 220, 70, 800, ["country", "vpn", "risk_score"], {"network_failure": 1.2}),
        _tool("velocity_rules", "Detect suspicious transaction velocity bursts.", 280, 90, 900, ["window", "count", "limit", "breach"], {"contract_drift": 1.1}),
        _tool("behavioral_anomaly", "Detect unusual behavior against user baseline.", 360, 110, 1100, ["anomaly_score", "signals", "explanation"], {"timeout": 1.2}),
        _tool("merchant_risk_api", "Score merchant fraud and chargeback propensity.", 300, 95, 1000, ["merchant_id", "risk_score", "risk_factors"], {"malformed_schema": 1.1}),
        _tool("chargeback_history", "Retrieve account chargeback history and trends.", 310, 100, 1000, ["chargebacks", "ratio", "period"], {"rate_limit": 1.2}),
        _tool("payment_network_status", "Get real-time processor and network incident status.", 180, 55, 700, ["network", "status", "degraded_regions"], {"network_failure": 1.3}),
        _tool("fraud_case_search", "Search historical fraud investigation cases.", 340, 100, 1050, ["cases", "query", "count"], {"rate_limit": 1.1}),
        _tool("sar_template_builder", "Build SAR draft with structured suspicious activity narrative.", 420, 130, 1250, ["title", "narrative", "evidence_refs"], {"malformed_schema": 1.3}),
        _tool("policy_clause_retriever", "Retrieve policy clause for compliance decisions.", 230, 75, 850, ["policy_id", "clause", "source"], {"contract_drift": 1.1}),
        _tool("risk_decision_engine", "Produce allow/challenge/deny decision with rationale.", 270, 85, 900, ["decision", "reason", "confidence"], {"contract_drift": 1.2}),
        _tool("limit_adjustment", "Recommend account limit changes based on risk profile.", 290, 95, 950, ["new_limit", "reason", "effective_date"], {"timeout": 1.1}),
        _tool("analyst_note_drafter", "Draft concise analyst notes for case timeline.", 250, 80, 850, ["summary", "next_steps", "risk_level"], {"malformed_schema": 1.2}),
    ]

    tasks = [
        _task("fin-001", "Check KYC status for customer C-191 and risk tier.", "kyc_profile_lookup", ["fraud_case_search", "policy_clause_retriever"], ["customer_id", "kyc_status", "risk_tier"]),
        _task("fin-002", "Run AML sanctions screening for new beneficiary.", "aml_screening", ["kyc_profile_lookup", "policy_clause_retriever"], ["match", "score", "watchlist", "reason"]),
        _task("fin-003", "Investigate suspicious ring using transaction flow graph.", "transaction_graph", ["behavioral_anomaly", "fraud_case_search"], ["nodes", "edges", "anomaly_score"]),
        _task("fin-004", "Assess trustworthiness of current login device.", "device_fingerprint", ["ip_geo_risk", "behavioral_anomaly"], ["device_id", "trust_score", "flags"]),
        _task("fin-005", "Evaluate IP geo risk for payment attempt from unknown country.", "ip_geo_risk", ["device_fingerprint", "behavioral_anomaly"], ["country", "vpn", "risk_score"]),
        _task("fin-006", "Detect if user exceeded transfer velocity rules today.", "velocity_rules", ["behavioral_anomaly", "risk_decision_engine"], ["window", "count", "limit", "breach"]),
        _task("fin-007", "Find unusual behavior relative to account baseline.", "behavioral_anomaly", ["velocity_rules", "risk_decision_engine"], ["anomaly_score", "signals", "explanation"]),
        _task("fin-008", "Score merchant M-44 for fraud exposure.", "merchant_risk_api", ["chargeback_history", "risk_decision_engine"], ["merchant_id", "risk_score", "risk_factors"]),
        _task("fin-009", "Get chargeback ratio trend for merchant M-44.", "chargeback_history", ["merchant_risk_api", "fraud_case_search"], ["chargebacks", "ratio", "period"]),
        _task("fin-010", "Are payment networks degraded in APAC right now?", "payment_network_status", ["analyst_note_drafter", "fraud_case_search"], ["network", "status", "degraded_regions"]),
        _task("fin-011", "Search past fraud cases matching mule account pattern.", "fraud_case_search", ["transaction_graph", "behavioral_anomaly"], ["cases", "query", "count"]),
        _task("fin-012", "Draft SAR narrative for suspicious layering activity.", "sar_template_builder", ["analyst_note_drafter", "fraud_case_search"], ["title", "narrative", "evidence_refs"]),
        _task("fin-013", "Retrieve compliance clause for enhanced due diligence step.", "policy_clause_retriever", ["kyc_profile_lookup", "aml_screening"], ["policy_id", "clause", "source"]),
        _task("fin-014", "Decide allow, challenge, or deny for this transfer request.", "risk_decision_engine", ["behavioral_anomaly", "velocity_rules"], ["decision", "reason", "confidence"]),
        _task("fin-015", "Recommend updated transfer limit for medium-risk account.", "limit_adjustment", ["risk_decision_engine", "kyc_profile_lookup"], ["new_limit", "reason", "effective_date"]),
        _task("fin-016", "Draft analyst note summarizing risk signals for review.", "analyst_note_drafter", ["risk_decision_engine", "fraud_case_search"], ["summary", "next_steps", "risk_level"]),
        _task("fin-017", "Need immediate deny/allow recommendation with rationale.", "risk_decision_engine", ["velocity_rules", "behavioral_anomaly"], ["decision", "reason", "confidence"]),
        _task("fin-018", "Build evidence-backed SAR section from known case links.", "sar_template_builder", ["fraud_case_search", "analyst_note_drafter"], ["title", "narrative", "evidence_refs"]),
    ]

    eval_cases = {
        "toolset_id": toolset_id,
        "cases": [
            {
                "case_id": f"{toolset_id}-{task['task_id']}",
                "task_id": task["task_id"],
                "question": task["user_query"],
                "expected_first_tool": task["primary_tool"],
                "expected_tool_sequence": [task["primary_tool"]],
                "acceptable_alternatives": task["fallback_tools"],
            }
            for task in tasks
        ],
    }
    workload = {"toolset_id": toolset_id, "tools": tools, "tasks": tasks}
    return workload, eval_cases


def main() -> int:
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    builders = [
        _build_customer_support_suite,
        _build_ecommerce_ops_suite,
        _build_fintech_risk_suite,
    ]

    manifest = []
    for build in builders:
        workload, eval_cases = build()
        toolset_id = str(workload["toolset_id"])

        workload_path = WORKLOAD_DIR / f"{toolset_id}.json"
        eval_path = EVAL_DIR / f"{toolset_id}_eval_cases.json"

        workload_path.write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8")
        eval_path.write_text(json.dumps(eval_cases, indent=2) + "\n", encoding="utf-8")

        manifest.append(
            {
                "toolset_id": toolset_id,
                "workload": str(workload_path.relative_to(ROOT)),
                "eval_cases": str(eval_path.relative_to(ROOT)),
                "tools": len(workload["tools"]),
                "tasks": len(workload["tasks"]),
            }
        )

    manifest_path = WORKLOAD_DIR / "manifest.json"
    manifest_path.write_text(json.dumps({"toolsets": manifest}, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {len(manifest)} toolsets")
    for item in manifest:
        print(
            f"- {item['toolset_id']}: tools={item['tools']} tasks={item['tasks']} "
            f"workload={item['workload']} eval={item['eval_cases']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
