from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .types import ToolDef, ToolExecutor


@dataclass
class ExecutableTool:
    definition: ToolDef
    state: dict[str, Any] = field(default_factory=dict)

    def execute(self, arguments: dict[str, Any], rng: random.Random) -> Any:
        if self.definition.executor is None:
            raise NotImplementedError(
                f"Tool {self.definition.name} has no executor registered"
            )
        return self.definition.executor(arguments, self.state, rng)


# ── Executor functions ───────────────────────────────────────────────


def _customer_lookup(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    customer_id = str(args.get("customer_id", ""))
    customers = {
        "C001": {"name": "Alice Chen", "tier": "premium", "email": "alice@example.com"},
        "C002": {"name": "Bob Kumar", "tier": "standard", "email": "bob@example.com"},
        "C003": {"name": "Carol Diaz", "tier": "standard", "email": "carol@example.com"},
        "C004": {"name": "Dave Patel", "tier": "premium", "email": "dave@example.com"},
    }
    record = customers.get(customer_id)
    if record is None:
        return {"found": False, "customer_id": customer_id}
    return {"found": True, "customer_id": customer_id, **record}


def _order_lookup(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    order_id = str(args.get("order_id", ""))
    orders = {
        "O1001": {"customer_id": "C001", "status": "shipped", "total": 149.99, "items": 3},
        "O1002": {"customer_id": "C002", "status": "processing", "total": 79.50, "items": 1},
        "O1003": {"customer_id": "C001", "status": "delivered", "total": 299.00, "items": 5},
        "O1004": {"customer_id": "C003", "status": "cancelled", "total": 49.99, "items": 2},
    }
    record = orders.get(order_id)
    if record is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order_id": order_id, **record}


def _return_status(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    return_id = str(args.get("return_id", ""))
    returns = {
        "R5001": {"order_id": "O1003", "status": "approved", "refund": 299.00},
        "R5002": {"order_id": "O1004", "status": "pending", "refund": None},
        "R5003": {"order_id": "O1002", "status": "rejected", "refund": None},
    }
    record = returns.get(return_id)
    if record is None:
        return {"found": False, "return_id": return_id}
    return {"found": True, "return_id": return_id, **record}


def _product_catalog(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    query = str(args.get("query", "")).lower()
    products = [
        {"sku": "SKU-A100", "name": "Wireless Headphones", "price": 79.99, "stock": 23},
        {"sku": "SKU-A200", "name": "USB-C Hub", "price": 49.99, "stock": 0},
        {"sku": "SKU-B100", "name": "Mechanical Keyboard", "price": 129.99, "stock": 15},
        {"sku": "SKU-B200", "name": "27\" Monitor", "price": 299.99, "stock": 8},
    ]
    matches = [p for p in products if query in p["name"].lower() or query in p["sku"].lower()]
    return {"products": matches, "total": len(matches)}


def _payment_charge(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    amount = float(args.get("amount", 0))
    method = str(args.get("method", "credit_card"))
    if amount <= 0:
        return {"success": False, "error": "invalid_amount", "charge_id": None}
    if amount > state.get("balance", 0):
        return {"success": False, "error": "insufficient_funds", "charge_id": None}
    state["balance"] = state.get("balance", 0) - amount
    charge_id = f"CHG-{_rng.randint(10000, 99999)}"
    return {"success": True, "charge_id": charge_id, "amount": amount, "method": method}


def _balance_check(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    account_id = str(args.get("account_id", ""))
    balances = {
        "ACC-100": {"balance": 1500.00, "currency": "USD"},
        "ACC-200": {"balance": 45.00, "currency": "USD"},
        "ACC-300": {"balance": 0.00, "currency": "USD"},
    }
    record = balances.get(account_id)
    if record is None:
        return {"found": False, "account_id": account_id}
    return {"found": True, "account_id": account_id, **record}


def _transaction_history(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    account_id = str(args.get("account_id", ""))
    limit = min(int(args.get("limit", 10)), 50)
    all_txns = {
        "ACC-100": [
            {"id": "TXN-01", "amount": -79.99, "merchant": "StoreA", "date": "2026-07-20"},
            {"id": "TXN-02", "amount": 500.00, "merchant": "Deposit", "date": "2026-07-15"},
            {"id": "TXN-03", "amount": -29.99, "merchant": "StoreB", "date": "2026-07-10"},
        ],
        "ACC-200": [
            {"id": "TXN-04", "amount": 200.00, "merchant": "Deposit", "date": "2026-07-22"},
            {"id": "TXN-05", "amount": -149.99, "merchant": "StoreC", "date": "2026-07-18"},
        ],
        "ACC-300": [],
    }
    txns = all_txns.get(account_id, [])
    return {"account_id": account_id, "transactions": txns[:limit], "total": len(txns)}


def _code_search(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    query = str(args.get("query", "")).lower()
    repo = str(args.get("repo", ""))
    limit = min(int(args.get("limit", 10)), 50)
    snippets = [
        {"file": "src/auth.py:42", "content": "def verify_token(token: str) -> bool:", "language": "python"},
        {"file": "src/db.py:108", "content": "async def query(sql: str, params: tuple)", "language": "python"},
        {"file": "pkg/handler.go:75", "content": "func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request)", "language": "go"},
        {"file": "lib/client.ts:23", "content": "export async function fetchUsers(): Promise<User[]>", "language": "typescript"},
    ]
    matches = [s for s in snippets if query in s["content"].lower()]
    if repo:
        matches = [s for s in matches if repo in s["file"]]
    return {"results": matches[:limit], "total": len(matches), "query": query}


def _issue_tracker(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    issue_id = str(args.get("issue_id", ""))
    issues = {
        "ISSUE-101": {"title": "Fix login timeout", "status": "open", "assignee": "alice", "priority": "high"},
        "ISSUE-102": {"title": "Add pagination to API", "status": "in_progress", "assignee": "bob", "priority": "medium"},
        "ISSUE-103": {"title": "Update dependencies", "status": "closed", "assignee": "carol", "priority": "low"},
    }
    record = issues.get(issue_id)
    if record is None:
        return {"found": False, "issue_id": issue_id}
    return {"found": True, "issue_id": issue_id, **record}


def _deploy_status(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    env = str(args.get("environment", "production"))
    deploys = {
        "production": {"version": "v2.4.1", "status": "healthy", "last_deploy": "2026-07-26T14:30:00Z", "commit": "abc123def"},
        "staging": {"version": "v2.5.0-rc1", "status": "healthy", "last_deploy": "2026-07-26T10:00:00Z", "commit": "def456ghi"},
    }
    record = deploys.get(env)
    if record is None:
        return {"found": False, "environment": env}
    return {"found": True, "environment": env, **record}


def _ticket_create(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    subject = str(args.get("subject", ""))
    priority = str(args.get("priority", "medium"))
    customer_id = str(args.get("customer_id", ""))
    ticket_id = f"TCK-{_rng.randint(10000, 99999)}"
    state.setdefault("tickets", {})[ticket_id] = {
        "subject": subject,
        "priority": priority,
        "customer_id": customer_id,
        "status": "open",
    }
    return {"ticket_id": ticket_id, "status": "open", "subject": subject, "priority": priority}


def _ticket_status(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    ticket_id = str(args.get("ticket_id", ""))
    if ticket_id in state.get("tickets", {}):
        return {"found": True, "ticket_id": ticket_id, **state["tickets"][ticket_id]}
    static = {
        "TCK-5001": {"subject": "Refund request", "status": "open", "priority": "high", "assignee": "alice"},
        "TCK-5002": {"subject": "Login issue", "status": "in_progress", "priority": "medium", "assignee": "bob"},
        "TCK-5003": {"subject": "Shipping delay", "status": "resolved", "priority": "low", "assignee": "carol"},
        "TCK-5004": {"subject": "Billing dispute", "status": "open", "priority": "critical", "assignee": "dave"},
    }
    record = static.get(ticket_id)
    if record is None:
        return {"found": False, "ticket_id": ticket_id}
    return {"found": True, "ticket_id": ticket_id, **record}


def _kb_search(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    query = str(args.get("query", "")).lower()
    articles = [
        {"id": "KB-001", "title": "How to reset your password", "category": "account", "confidence": 0.95},
        {"id": "KB-002", "title": "Return policy and timeline", "category": "orders", "confidence": 0.88},
        {"id": "KB-003", "title": "Shipping rates by region", "category": "shipping", "confidence": 0.92},
        {"id": "KB-004", "title": "Refund processing time", "category": "billing", "confidence": 0.90},
        {"id": "KB-005", "title": "Account deactivation policy", "category": "account", "confidence": 0.85},
        {"id": "KB-006", "title": "How to track your order", "category": "orders", "confidence": 0.97},
    ]
    matches = [a for a in articles if query in a["title"].lower() or query in a["category"]]
    return {"results": matches, "total": len(matches), "query": query}


def _refund_eligibility(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    order_id = str(args.get("order_id", ""))
    rules = {
        "O1001": {"eligible": True, "reason": "Within 30-day window", "max_refund": 149.99},
        "O1002": {"eligible": True, "reason": "Unshipped order", "max_refund": 79.50},
        "O1003": {"eligible": False, "reason": "Past 60-day return window", "max_refund": 0},
        "O1004": {"eligible": False, "reason": "Order already cancelled", "max_refund": 0},
    }
    record = rules.get(order_id)
    if record is None:
        return {"eligible": False, "reason": "Order not found", "order_id": order_id, "max_refund": 0}
    return {"order_id": order_id, **record}


def _inventory_check(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    sku = str(args.get("sku", ""))
    warehouse = str(args.get("warehouse", ""))
    inventory = {
        "SKU-A100": {"warehouse": "DAL-01", "available": 23, "reserved": 2, "location": "Dallas, TX"},
        "SKU-A200": {"warehouse": "DAL-01", "available": 0, "reserved": 0, "location": "Dallas, TX"},
        "SKU-B100": {"warehouse": "NYC-01", "available": 15, "reserved": 1, "location": "New York, NY"},
        "SKU-B200": {"warehouse": "SFO-01", "available": 8, "reserved": 3, "location": "San Francisco, CA"},
        "SKU-C100": {"warehouse": "DAL-01", "available": 42, "reserved": 0, "location": "Dallas, TX"},
    }
    record = inventory.get(sku)
    if record is None:
        return {"found": False, "sku": sku}
    if warehouse and warehouse.upper() != record["warehouse"]:
        return {"found": True, "sku": sku, "available": 0, "reserved": 0, "location": warehouse, "note": "Different warehouse"}
    return {"found": True, "sku": sku, **record}


def _shipping_quote(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    destination = str(args.get("destination", "")).upper()
    weight_kg = float(args.get("weight_kg", 1.0))
    rates = {
        "BOSTON": {"carrier": "FedEx", "cost": 12.99, "eta_days": 3, "service": "Ground"},
        "DALLAS": {"carrier": "UPS", "cost": 8.50, "eta_days": 2, "service": "Ground"},
        "SF": {"carrier": "USPS", "cost": 15.75, "eta_days": 5, "service": "Priority"},
        "NYC": {"carrier": "FedEx", "cost": 11.25, "eta_days": 3, "service": "Ground"},
        "LONDON": {"carrier": "DHL", "cost": 45.00, "eta_days": 7, "service": "International"},
    }
    base = rates.get(destination)
    if base is None:
        return {"carrier": "Unknown", "cost": 20.00, "eta_days": 5, "service": "Standard", "destination": destination}
    cost = round(base["cost"] * (1.0 + 0.5 * max(0, weight_kg - 1)), 2)
    return {"carrier": base["carrier"], "cost": cost, "eta_days": base["eta_days"], "service": base["service"], "destination": destination}


def _payment_verify(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    order_id = str(args.get("order_id", ""))
    payments = {
        "O1001": {"verified": True, "method": "credit_card", "amount": 149.99, "auth_id": "AUTH-7841"},
        "O1002": {"verified": True, "method": "paypal", "amount": 79.50, "auth_id": "AUTH-7842"},
        "O1003": {"verified": True, "method": "credit_card", "amount": 299.00, "auth_id": "AUTH-7843"},
        "O1004": {"verified": False, "method": "credit_card", "amount": 49.99, "auth_id": None, "reason": "cancelled"},
    }
    record = payments.get(order_id)
    if record is None:
        return {"verified": False, "order_id": order_id, "reason": "No payment record found"}
    return {"order_id": order_id, **record}


def _promotion_check(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    code = str(args.get("code", "")).upper()
    order_total = float(args.get("order_total", 0))
    promos = {
        "SUMMER20": {"valid": True, "discount_pct": 20, "min_order": 50, "description": "Summer sale 20% off"},
        "WELCOME10": {"valid": True, "discount_pct": 10, "min_order": 25, "description": "Welcome 10% off"},
        "FREESHIP": {"valid": True, "discount_pct": 0, "min_order": 75, "description": "Free shipping"},
        "VIP50": {"valid": True, "discount_pct": 50, "min_order": 200, "description": "VIP member 50% off"},
        "EXPIRED": {"valid": False, "discount_pct": 30, "min_order": 50, "description": "Expired promotion"},
    }
    promo = promos.get(code)
    if promo is None:
        return {"valid": False, "code": code, "reason": "Unknown promotion code"}
    if order_total < promo["min_order"]:
        return {"valid": True, "code": code, "discount_pct": promo["discount_pct"],
                "applicable": False, "reason": f"Order below minimum {promo['min_order']}"}
    discount = round(order_total * promo["discount_pct"] / 100, 2)
    return {"valid": promo["valid"], "code": code, "discount_pct": promo["discount_pct"],
            "applicable": promo["valid"] and order_total >= promo["min_order"],
            "discount_amount": discount if promo["valid"] else 0, "description": promo["description"]}


def _fraud_check(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    transaction_id = str(args.get("transaction_id", ""))
    amount = float(args.get("amount", 0))
    scores = {
        "TXN-01": {"risk_score": 0.15, "flags": [], "recommendation": "approve"},
        "TXN-02": {"risk_score": 0.08, "flags": [], "recommendation": "approve"},
        "TXN-03": {"risk_score": 0.72, "flags": ["unusual_location", "high_amount"], "recommendation": "review"},
        "TXN-04": {"risk_score": 0.95, "flags": ["known_fraud_pattern", "velocity_check_failed"], "recommendation": "block"},
    }
    if transaction_id in scores:
        return {"transaction_id": transaction_id, **scores[transaction_id]}
    risk = min(0.95, max(0.05, amount / 500))
    return {"transaction_id": transaction_id, "risk_score": round(risk, 2),
            "flags": ["high_amount"] if amount > 500 else [], "recommendation": "review" if risk > 0.5 else "approve"}


def _loan_status(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    loan_id = str(args.get("loan_id", ""))
    loans = {
        "LOAN-001": {"amount": 15000, "status": "approved", "rate_pct": 5.9, "term_months": 36, "monthly": 455.12},
        "LOAN-002": {"amount": 5000, "status": "pending", "rate_pct": None, "term_months": 12, "monthly": None},
        "LOAN-003": {"amount": 25000, "status": "rejected", "rate_pct": None, "term_months": 48, "monthly": None, "reason": "Credit score below threshold"},
        "LOAN-004": {"amount": 8000, "status": "active", "rate_pct": 7.2, "term_months": 24, "monthly": 359.67, "remaining": 6400.00},
    }
    record = loans.get(loan_id)
    if record is None:
        return {"found": False, "loan_id": loan_id}
    return {"found": True, "loan_id": loan_id, **record}


def _fund_transfer(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    from_account = str(args.get("from_account", ""))
    to_account = str(args.get("to_account", ""))
    amount = float(args.get("amount", 0))
    balances = {"ACC-100": 1500.00, "ACC-200": 45.00, "ACC-300": 0.00}
    if from_account not in balances:
        return {"success": False, "error": "Source account not found", "transfer_id": None}
    if to_account not in balances:
        return {"success": False, "error": "Destination account not found", "transfer_id": None}
    if amount <= 0:
        return {"success": False, "error": "Invalid amount", "transfer_id": None}
    if balances[from_account] < amount:
        return {"success": False, "error": "Insufficient funds", "transfer_id": None}
    balances[from_account] -= amount
    balances[to_account] += amount
    transfer_id = f"XFR-{_rng.randint(10000, 99999)}"
    return {"success": True, "transfer_id": transfer_id, "amount": amount, "from_account": from_account, "to_account": to_account}


def _build_status(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    build_id = str(args.get("build_id", ""))
    project = str(args.get("project", ""))
    builds = {
        "BUILD-401": {"project": "frontend", "status": "passed", "branch": "main", "duration_s": 245, "commit": "abc123def"},
        "BUILD-402": {"project": "backend", "status": "failed", "branch": "feat/payments", "duration_s": 312, "commit": "def456ghi", "error": "3 tests failed in payment_service"},
        "BUILD-403": {"project": "mobile", "status": "running", "branch": "main", "duration_s": None, "commit": "ghi789jkl"},
        "BUILD-404": {"project": "frontend", "status": "passed", "branch": "fix/login", "duration_s": 198, "commit": "jkl012mno"},
    }
    for bid, b in builds.items():
        if bid == build_id or (project and b["project"] == project):
            return {"build_id": bid, **b}
    return {"found": False, "build_id": build_id, "project": project}


def _log_search(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    query = str(args.get("query", "")).lower()
    service = str(args.get("service", ""))
    limit = min(int(args.get("limit", 20)), 100)
    logs = [
        {"timestamp": "2026-07-27T10:30:00Z", "level": "ERROR", "service": "auth", "message": "Rate limit exceeded for IP 192.168.1.1"},
        {"timestamp": "2026-07-27T10:31:00Z", "level": "ERROR", "service": "db", "message": "Connection pool exhausted"},
        {"timestamp": "2026-07-27T10:32:00Z", "level": "WARN", "service": "auth", "message": "Slow query detected: select * from users"},
        {"timestamp": "2026-07-27T10:33:00Z", "level": "ERROR", "service": "api", "message": "500 Internal Server Error on /api/orders"},
        {"timestamp": "2026-07-27T10:34:00Z", "level": "INFO", "service": "deploy", "message": "Deployment v2.4.1 completed successfully"},
        {"timestamp": "2026-07-27T10:35:00Z", "level": "ERROR", "service": "payment", "message": "Timeout processing charge TXN-03"},
    ]
    matches = [e for e in logs if query in e["message"].lower() or query in e["level"].lower()]
    if service:
        matches = [e for e in matches if e["service"] == service]
    return {"entries": matches[:limit], "total": len(matches), "query": query}


def _dependency_check(
    args: dict[str, Any], state: dict[str, Any], _rng: random.Random
) -> dict[str, Any]:
    package = str(args.get("package", ""))
    deps = {
        "react": {"version": "18.2.0", "latest": "19.1.0", "vulnerabilities": 0, "license": "MIT"},
        "lodash": {"version": "4.17.20", "latest": "4.17.21", "vulnerabilities": 1, "license": "MIT", "cve": "CVE-2021-23337"},
        "express": {"version": "4.18.2", "latest": "5.0.0", "vulnerabilities": 2, "license": "MIT", "cves": ["CVE-2024-29041", "CVE-2024-10468"]},
        "next": {"version": "14.2.5", "latest": "15.1.0", "vulnerabilities": 0, "license": "MIT"},
        "axios": {"version": "1.7.2", "latest": "1.7.9", "vulnerabilities": 0, "license": "MIT"},
    }
    record = deps.get(package.lower())
    if record is None:
        return {"found": False, "package": package}
    return {"found": True, "package": package, **record}


# ── Tool registry ────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolDef] = {
    # Original tools (10)
    "customer_lookup": ToolDef(
        name="customer_lookup",
        description="Look up a customer by ID. Returns name, tier, and email.",
        input_schema={
            "type": "object",
            "properties": {"customer_id": {"type": "string", "description": "The customer ID, e.g. C001"}},
            "required": ["customer_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"}, "customer_id": {"type": "string"},
                "name": {"type": "string"}, "tier": {"type": "string"}, "email": {"type": "string"},
            },
        },
        executor=_customer_lookup,
    ),
    "order_lookup": ToolDef(
        name="order_lookup",
        description="Look up an order by ID. Returns customer, status, total, and item count.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order ID, e.g. O1001"}},
            "required": ["order_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"}, "order_id": {"type": "string"},
                "customer_id": {"type": "string"}, "status": {"type": "string"},
                "total": {"type": "number"}, "items": {"type": "integer"},
            },
        },
        executor=_order_lookup,
    ),
    "return_status": ToolDef(
        name="return_status",
        description="Check the status of a return by return ID.",
        input_schema={
            "type": "object",
            "properties": {"return_id": {"type": "string", "description": "The return ID, e.g. R5001"}},
            "required": ["return_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"}, "return_id": {"type": "string"},
                "order_id": {"type": "string"}, "status": {"type": "string"}, "refund": {"type": "number"},
            },
        },
        executor=_return_status,
    ),
    "product_catalog": ToolDef(
        name="product_catalog",
        description="Search the product catalog by name or SKU.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query for product name or SKU"}},
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "products": {"type": "array", "items": {"type": "object", "properties": {
                    "sku": {"type": "string"}, "name": {"type": "string"},
                    "price": {"type": "number"}, "stock": {"type": "integer"},
                }}},
                "total": {"type": "integer"},
            },
        },
        executor=_product_catalog,
    ),
    "payment_charge": ToolDef(
        name="payment_charge",
        description="Charge a payment from an account. Deducts from balance.",
        input_schema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to charge"},
                "method": {"type": "string", "description": "Payment method"},
            },
            "required": ["amount", "method"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "success": {"type": "boolean"}, "charge_id": {"type": "string"},
                "amount": {"type": "number"}, "method": {"type": "string"}, "error": {"type": "string"},
            },
        },
        executor=_payment_charge,
    ),
    "balance_check": ToolDef(
        name="balance_check",
        description="Check the balance of a financial account.",
        input_schema={
            "type": "object",
            "properties": {"account_id": {"type": "string", "description": "The account ID, e.g. ACC-100"}},
            "required": ["account_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"}, "account_id": {"type": "string"},
                "balance": {"type": "number"}, "currency": {"type": "string"},
            },
        },
        executor=_balance_check,
    ),
    "transaction_history": ToolDef(
        name="transaction_history",
        description="Get recent transactions for a financial account.",
        input_schema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "The account ID, e.g. ACC-100"},
                "limit": {"type": "integer", "description": "Max transactions to return, default 10"},
            },
            "required": ["account_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "transactions": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "amount": {"type": "number"},
                    "merchant": {"type": "string"}, "date": {"type": "string"},
                }}},
                "total": {"type": "integer"},
            },
        },
        executor=_transaction_history,
    ),
    "code_search": ToolDef(
        name="code_search",
        description="Search code across repositories by content.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Code fragment or keyword to search for"},
                "repo": {"type": "string", "description": "Optional repository filter"},
                "limit": {"type": "integer", "description": "Max results, default 10"},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array", "items": {"type": "object", "properties": {
                    "file": {"type": "string"}, "content": {"type": "string"}, "language": {"type": "string"},
                }}},
                "total": {"type": "integer"}, "query": {"type": "string"},
            },
        },
        executor=_code_search,
    ),
    "issue_tracker": ToolDef(
        name="issue_tracker",
        description="Look up an issue by ID. Returns title, status, assignee, and priority.",
        input_schema={
            "type": "object",
            "properties": {"issue_id": {"type": "string", "description": "The issue ID, e.g. ISSUE-101"}},
            "required": ["issue_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"}, "issue_id": {"type": "string"},
                "title": {"type": "string"}, "status": {"type": "string"},
                "assignee": {"type": "string"}, "priority": {"type": "string"},
            },
        },
        executor=_issue_tracker,
    ),
    "deploy_status": ToolDef(
        name="deploy_status",
        description="Check deployment status for an environment.",
        input_schema={
            "type": "object",
            "properties": {"environment": {"type": "string", "description": "Environment name, e.g. production or staging"}},
            "required": ["environment"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "found": {"type": "boolean"}, "environment": {"type": "string"},
                "version": {"type": "string"}, "status": {"type": "string"},
                "last_deploy": {"type": "string"}, "commit": {"type": "string"},
            },
        },
        executor=_deploy_status,
    ),
    # ── Customer support additions ──
    "ticket_create": ToolDef(
        name="ticket_create",
        description="Create a new support ticket with subject and priority.",
        input_schema={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Brief description of the issue"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "customer_id": {"type": "string", "description": "Customer ID for the ticket"},
            },
            "required": ["subject", "priority", "customer_id"],
        },
        output_schema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "status": {"type": "string"},
                          "subject": {"type": "string"}, "priority": {"type": "string"}},
        },
        executor=_ticket_create,
    ),
    "ticket_status": ToolDef(
        name="ticket_status",
        description="Check the status of an existing support ticket.",
        input_schema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string", "description": "The ticket ID, e.g. TCK-5001"}},
            "required": ["ticket_id"],
        },
        output_schema={
            "type": "object",
            "properties": {"found": {"type": "boolean"}, "ticket_id": {"type": "string"},
                          "subject": {"type": "string"}, "status": {"type": "string"},
                          "priority": {"type": "string"}, "assignee": {"type": "string"}},
        },
        executor=_ticket_status,
    ),
    "kb_search": ToolDef(
        name="kb_search",
        description="Search the knowledge base for help articles relevant to a query.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query for knowledge base articles"}},
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "title": {"type": "string"},
                    "category": {"type": "string"}, "confidence": {"type": "number"},
                }}},
                "total": {"type": "integer"}, "query": {"type": "string"},
            },
        },
        executor=_kb_search,
    ),
    "refund_eligibility": ToolDef(
        name="refund_eligibility",
        description="Check if an order is eligible for a refund and the maximum refund amount.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order ID to check, e.g. O1001"}},
            "required": ["order_id"],
        },
        output_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "eligible": {"type": "boolean"},
                          "reason": {"type": "string"}, "max_refund": {"type": "number"}},
        },
        executor=_refund_eligibility,
    ),
    # ── Ecommerce additions ──
    "inventory_check": ToolDef(
        name="inventory_check",
        description="Check real-time inventory availability for a product SKU at a warehouse.",
        input_schema={
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Product SKU, e.g. SKU-A100"},
                "warehouse": {"type": "string", "description": "Optional warehouse code filter, e.g. DAL-01"},
            },
            "required": ["sku"],
        },
        output_schema={
            "type": "object",
            "properties": {"found": {"type": "boolean"}, "sku": {"type": "string"},
                          "warehouse": {"type": "string"}, "available": {"type": "integer"},
                          "reserved": {"type": "integer"}, "location": {"type": "string"}},
        },
        executor=_inventory_check,
    ),
    "shipping_quote": ToolDef(
        name="shipping_quote",
        description="Get a shipping cost estimate and delivery ETA for a destination.",
        input_schema={
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "Destination city, e.g. BOSTON, NYC, SF"},
                "weight_kg": {"type": "number", "description": "Package weight in kg, default 1.0"},
            },
            "required": ["destination"],
        },
        output_schema={
            "type": "object",
            "properties": {"carrier": {"type": "string"}, "cost": {"type": "number"},
                          "eta_days": {"type": "integer"}, "service": {"type": "string"},
                          "destination": {"type": "string"}},
        },
        executor=_shipping_quote,
    ),
    "payment_verify": ToolDef(
        name="payment_verify",
        description="Verify payment status for an order. Returns method, amount, and auth ID.",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order ID, e.g. O1001"}},
            "required": ["order_id"],
        },
        output_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "verified": {"type": "boolean"},
                          "method": {"type": "string"}, "amount": {"type": "number"},
                          "auth_id": {"type": "string"}, "reason": {"type": "string"}},
        },
        executor=_payment_verify,
    ),
    "promotion_check": ToolDef(
        name="promotion_check",
        description="Validate a promotion code and compute discount for an order total.",
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Promotion code, e.g. SUMMER20"},
                "order_total": {"type": "number", "description": "Current order total before discount"},
            },
            "required": ["code", "order_total"],
        },
        output_schema={
            "type": "object",
            "properties": {"valid": {"type": "boolean"}, "code": {"type": "string"},
                          "discount_pct": {"type": "number"}, "applicable": {"type": "boolean"},
                          "discount_amount": {"type": "number"}, "description": {"type": "string"}},
        },
        executor=_promotion_check,
    ),
    # ── Fintech additions ──
    "fraud_check": ToolDef(
        name="fraud_check",
        description="Run fraud detection on a transaction. Returns risk score and recommendation.",
        input_schema={
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string", "description": "Transaction ID, e.g. TXN-01"},
                "amount": {"type": "number", "description": "Transaction amount"},
            },
            "required": ["transaction_id", "amount"],
        },
        output_schema={
            "type": "object",
            "properties": {"transaction_id": {"type": "string"}, "risk_score": {"type": "number"},
                          "flags": {"type": "array", "items": {"type": "string"}},
                          "recommendation": {"type": "string", "enum": ["approve", "review", "block"]}},
        },
        executor=_fraud_check,
    ),
    "loan_status": ToolDef(
        name="loan_status",
        description="Check the status and details of a loan application.",
        input_schema={
            "type": "object",
            "properties": {"loan_id": {"type": "string", "description": "Loan application ID, e.g. LOAN-001"}},
            "required": ["loan_id"],
        },
        output_schema={
            "type": "object",
            "properties": {"found": {"type": "boolean"}, "loan_id": {"type": "string"},
                          "amount": {"type": "number"}, "status": {"type": "string"},
                          "rate_pct": {"type": "number"}, "term_months": {"type": "integer"},
                          "monthly": {"type": "number"}, "reason": {"type": "string"},
                          "remaining": {"type": "number"}},
        },
        executor=_loan_status,
    ),
    "fund_transfer": ToolDef(
        name="fund_transfer",
        description="Transfer funds between two accounts. Source must have sufficient balance.",
        input_schema={
            "type": "object",
            "properties": {
                "from_account": {"type": "string", "description": "Source account ID, e.g. ACC-100"},
                "to_account": {"type": "string", "description": "Destination account ID, e.g. ACC-200"},
                "amount": {"type": "number", "description": "Amount to transfer"},
            },
            "required": ["from_account", "to_account", "amount"],
        },
        output_schema={
            "type": "object",
            "properties": {"success": {"type": "boolean"}, "transfer_id": {"type": "string"},
                          "amount": {"type": "number"}, "from_account": {"type": "string"},
                          "to_account": {"type": "string"}, "error": {"type": "string"}},
        },
        executor=_fund_transfer,
    ),
    # ── Developer tools additions ──
    "build_status": ToolDef(
        name="build_status",
        description="Check CI/CD build status by build ID or project name.",
        input_schema={
            "type": "object",
            "properties": {
                "build_id": {"type": "string", "description": "Build ID, e.g. BUILD-401"},
                "project": {"type": "string", "description": "Project name, e.g. frontend, backend"},
            },
            "required": [],
        },
        output_schema={
            "type": "object",
            "properties": {"build_id": {"type": "string"}, "project": {"type": "string"},
                          "status": {"type": "string", "enum": ["passed", "failed", "running"]},
                          "branch": {"type": "string"}, "duration_s": {"type": "number"},
                          "commit": {"type": "string"}, "error": {"type": "string"}},
        },
        executor=_build_status,
    ),
    "log_search": ToolDef(
        name="log_search",
        description="Search application logs by keyword and optional service filter.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword to search for in log messages"},
                "service": {"type": "string", "description": "Optional service name filter, e.g. auth, api"},
                "limit": {"type": "integer", "description": "Max entries to return, default 20"},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "entries": {"type": "array", "items": {"type": "object", "properties": {
                    "timestamp": {"type": "string"}, "level": {"type": "string"},
                    "service": {"type": "string"}, "message": {"type": "string"},
                }}},
                "total": {"type": "integer"}, "query": {"type": "string"},
            },
        },
        executor=_log_search,
    ),
    "dependency_check": ToolDef(
        name="dependency_check",
        description="Check a package dependency for version, latest, vulnerabilities, and license.",
        input_schema={
            "type": "object",
            "properties": {"package": {"type": "string", "description": "Package name, e.g. react, lodash, express"}},
            "required": ["package"],
        },
        output_schema={
            "type": "object",
            "properties": {"found": {"type": "boolean"}, "package": {"type": "string"},
                          "version": {"type": "string"}, "latest": {"type": "string"},
                          "vulnerabilities": {"type": "integer"}, "license": {"type": "string"},
                          "cve": {"type": "string"}, "cves": {"type": "array", "items": {"type": "string"}}},
        },
        executor=_dependency_check,
    ),
}


def build_executable_tools(names: list[str], shared_state: dict[str, Any] | None = None) -> list[ExecutableTool]:
    tools: list[ExecutableTool] = []
    for name in names:
        if name not in TOOL_REGISTRY:
            raise KeyError(f"Unknown tool: {name}")
        tools.append(ExecutableTool(definition=TOOL_REGISTRY[name], state=dict(shared_state or {})))
    return tools
