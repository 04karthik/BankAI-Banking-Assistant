"""Deterministic supervisor and tool agents for the BankAI demo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bankai import storage
from bankai.llm import (
    LLMError,
    RoutingPlan,
    classify_question,
    load_llm_settings,
)


@dataclass
class AssistantResult:
    answer: str
    trace: list[dict[str, str]]
    mode: str
    mode_detail: str


def _trace(trace: list[dict[str, str]], agent: str, detail: str) -> None:
    trace.append({"agent": agent, "status": "Complete", "detail": detail})


def banking_agent(
    customer_id: str,
    message: str,
    db_path: str = str(storage.DATABASE_PATH),
    requested_actions: tuple[str, ...] | None = None,
) -> str:
    customer = storage.get_customer(customer_id, db_path)
    account = storage.get_account(customer_id, db_path)
    if not customer or not account:
        return "I couldn't find a banking profile for the selected demo customer."

    text = message.lower()
    if requested_actions is None:
        wants_transactions = any(
            phrase in text
            for phrase in (
                "transaction",
                "statement",
                "recent activity",
                "recent charge",
                "last purchase",
            )
        )
        wants_balance = any(
            phrase in text
            for phrase in ("balance", "account details", "account number", "checking")
        )
    else:
        wants_transactions = "transactions" in requested_actions
        wants_balance = "balance" in requested_actions
    sections: list[str] = []
    if wants_transactions:
        match = re.search(
            r"\b(\d{1,2})\s+(?:(?:most\s+)?(?:recent|latest|last)\s+)?transactions?\b",
            text,
        )
        if not match:
            match = re.search(r"\b(?:last|recent|show)(?:\s+my)?\s+(\d{1,2})\b", text)
        limit = int(match.group(1)) if match else 5
        transactions = storage.get_transactions(customer_id, limit, db_path)
        if not transactions:
            sections.append("No transactions were found for this demo account.")
        else:
            lines = [
                f"Here are the {len(transactions)} most recent transactions for "
                f"**{customer['name']}**:"
            ]
            for transaction in transactions:
                amount = float(transaction["amount"])
                sign = "+" if amount >= 0 else "−"
                lines.append(
                    f"- {transaction['posted_at']} · **{transaction['merchant']}** "
                    f"({transaction['category']}) — {sign}${abs(amount):,.2f} "
                    f"· {transaction['status']}"
                )
            sections.append("\n".join(lines))

    if wants_balance or not wants_transactions:
        sections.append(
            f"**{customer['name']} · {account['account_type']}**\n\n"
            f"- Available balance: **${account['balance']:,.2f} {account['currency']}**\n"
            f"- Account ending: **{account['account_id'][-4:]}**\n"
            f"- Account ID: `{account['account_id']}`"
        )
    return "\n\n".join(sections)


def _amount_for_intent(message: str, intent: str, default: float) -> float:
    amount_matches = list(
        re.finditer(
            r"\$\s*(\d[\d,]*(?:\.\d{1,2})?)|"
            r"\b(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\b",
            message,
        )
    )
    if not amount_matches:
        return default

    markers = (
        ("risk", "fraud", "suspicious", "purchase", "transaction", "payment", "charge")
        if intent == "fraud"
        else ("loan", "borrow", "eligib", "emi", "installment")
    )
    marker_positions = [
        match.start()
        for marker in markers
        for match in re.finditer(re.escape(marker), message, flags=re.IGNORECASE)
    ]
    best_match = min(
        amount_matches,
        key=lambda match: min(
            (abs(match.start() - position) for position in marker_positions),
            default=10000,
        )
        - (2 if match.group(0).startswith("$") else 0),
    )
    raw_amount = best_match.group(1) or best_match.group(2)
    return float(raw_amount.replace(",", ""))


def _merchant_context(message: str) -> str:
    return message.lower()


def fraud_agent(message: str) -> tuple[str, int, str]:
    amount = _amount_for_intent(message, "fraud", 120.0)
    context = _merchant_context(message)
    risk = 8
    signals: list[str] = []
    if amount >= 500:
        risk += 18
        signals.append("larger-than-usual amount")
    if amount >= 1500:
        risk += 18
    if amount >= 5000:
        risk += 20
    if any(
        keyword in context
        for keyword in ("international", "overseas", "singapore", "lagos", "london", "tokyo")
    ):
        risk += 24
        signals.append("international location")
    if any(
        keyword in context
        for keyword in ("crypto", "wire transfer", "gift card", "cash advance")
    ):
        risk += 24
        signals.append("higher-risk transaction type")
    if "online" in context or "web purchase" in context:
        risk += 10
        signals.append("card-not-present purchase")
    if any(keyword in context for keyword in ("3am", "4am", "midnight", "late night")):
        risk += 10
        signals.append("unusual hour")

    score = min(risk, 100)
    category = "Low" if score < 30 else "Moderate" if score < 60 else "High"
    signal_text = ", ".join(signals) if signals else "no elevated signals in the supplied details"
    result = (
        f"**Simulated transaction risk: {category} ({score}/100)**\n\n"
        f"Amount assessed: **${amount:,.2f}**. Signals: {signal_text}. "
        "This score is a deterministic demo, not a bank fraud decision."
    )
    return result, score, category


def calculate_emi(principal: float, annual_rate_percent: float, months: int) -> float:
    if principal <= 0:
        raise ValueError("Loan amount must be greater than zero.")
    if months <= 0:
        raise ValueError("Loan term must be at least one month.")
    monthly_rate = annual_rate_percent / 100 / 12
    if monthly_rate == 0:
        return principal / months
    growth = (1 + monthly_rate) ** months
    return principal * monthly_rate * growth / (growth - 1)


def loan_agent(customer_id: str, message: str, db_path: str = str(storage.DATABASE_PATH)) -> str:
    customer = storage.get_customer(customer_id, db_path)
    if not customer:
        return "I couldn't find a loan profile for the selected demo customer."

    requested_amount = _amount_for_intent(message, "loan", 20000.0)
    term_match = re.search(r"\b(\d{1,3})\s*(?:months?|mos?)\b", message, flags=re.IGNORECASE)
    months = int(term_match.group(1)) if term_match else 36
    if months < 1 or months > 120:
        months = min(max(months, 1), 120)

    annual_rate = 11.5
    emi = calculate_emi(requested_amount, annual_rate, months)
    monthly_income = float(customer["monthly_income"])
    monthly_debt = float(customer["monthly_debt"])
    score = int(customer["credit_score"])
    affordable_payment = max(0.0, monthly_income * 0.35 - monthly_debt)
    amount_within_policy = requested_amount <= monthly_income * 12
    eligible = score >= 600 and emi <= affordable_payment and amount_within_policy

    if eligible:
        decision = "**Preliminary result: likely eligible** under this demo's simplified rules."
    else:
        reasons = []
        if score < 600:
            reasons.append("simulated credit score is below 600")
        if emi > affordable_payment:
            reasons.append("estimated EMI exceeds the demo's affordability limit")
        if not amount_within_policy:
            reasons.append("requested amount is above 12× simulated monthly income")
        decision = (
            "**Preliminary result: not eligible under this demo's rules.** "
            "Reason: " + "; ".join(reasons) + "."
        )

    return (
        f"{decision}\n\n"
        f"- Requested amount: **${requested_amount:,.2f}**\n"
        f"- Term: **{months} months** at a simulated **{annual_rate:.1f}%** annual rate\n"
        f"- Estimated monthly EMI: **${emi:,.2f}**\n"
        f"- Simulated monthly income / existing debt: "
        f"**${monthly_income:,.2f} / ${monthly_debt:,.2f}**\n"
        f"- Simulated credit score: **{score}**\n\n"
        "This is an educational estimate, not a lending decision or offer."
    )


def _mock_routing_plan(message: str) -> RoutingPlan:
    """Preserve the original keyword router when live LLM routing is unavailable."""
    text = message.lower()
    wants_fraud = any(
        keyword in text for keyword in ("fraud", "risk", "suspicious", "safe purchase")
    )
    wants_loan = any(
        keyword in text
        for keyword in ("loan", "borrow", "emi", "eligibility", "eligible", "installment")
    )
    wants_banking = any(
        keyword in text
        for keyword in (
            "balance",
            "account",
            "transaction history",
            "recent transaction",
            "recent activity",
            "statement",
            "last purchase",
        )
    )

    selected: list[str] = []
    if wants_banking:
        selected.append("banking")
    if wants_fraud:
        selected.append("fraud")
    if wants_loan:
        selected.append("loan")

    banking_actions: list[str] = []
    if wants_banking:
        wants_transactions = any(
            phrase in text
            for phrase in (
                "transaction",
                "statement",
                "recent activity",
                "recent charge",
                "last purchase",
            )
        )
        wants_balance = any(
            phrase in text for phrase in ("balance", "account details", "account number", "checking")
        )
        if wants_transactions:
            banking_actions.append("transactions")
        if wants_balance or not wants_transactions:
            banking_actions.append("balance")
    return RoutingPlan(tuple(selected), tuple(banking_actions))


def supervisor_agent(
    customer_id: str, message: str, db_path: str = str(storage.DATABASE_PATH)
) -> AssistantResult:
    """Route a question using a validated LLM plan or the original mock router."""
    trace: list[dict[str, str]] = []
    settings = load_llm_settings()
    mode = "MOCK MODE"
    if settings.ready:
        try:
            plan = classify_question(message, settings)
            mode = "LIVE LLM MODE"
            mode_detail = (
                f"The {settings.provider} intent plan passed validation; "
                "balances and specialist calculations stayed local."
            )
        except LLMError as error:
            plan = _mock_routing_plan(message)
            mode_detail = f"LLM routing fallback: {error}"
    else:
        plan = _mock_routing_plan(message)
        if settings.has_api_key and settings.config_error:
            mode_detail = (
                f"LLM configuration issue: {settings.config_error} "
                "Deterministic routing was used."
            )
        else:
            mode_detail = (
                "No API key configured; deterministic keyword routing was used."
            )

    agent_labels = {
        "banking": "Banking Agent",
        "fraud": "Fraud Agent",
        "loan": "Loan Agent",
    }
    selected = [agent_labels[agent] for agent in plan.agents]
    _trace(
        trace,
        "Supervisor Agent",
        f"{mode}: {mode_detail} "
        f"Routed to {', '.join(selected)}."
        if selected
        else f"{mode}: {mode_detail} No specialist matched; returned the supported demo capabilities.",
    )

    responses: list[str] = []
    for agent_id in plan.agents:
        agent_name = agent_labels[agent_id]
        if agent_id == "banking":
            responses.append(
                banking_agent(
                    customer_id,
                    message,
                    db_path,
                    requested_actions=plan.banking_actions,
                )
            )
            _trace(trace, agent_name, "Read account and/or transaction data from SQLite.")
        elif agent_id == "fraud":
            result, risk_score, category = fraud_agent(message)
            responses.append(result)
            _trace(
                trace,
                agent_name,
                f"Computed a deterministic simulated risk score ({category}, {risk_score}/100).",
            )
        elif agent_id == "loan":
            responses.append(loan_agent(customer_id, message, db_path))
            _trace(trace, agent_name, "Checked the simulated customer profile and calculated EMI.")

    if not responses:
        responses.append(
            "I can help with account balances, recent transactions, simulated transaction "
            "risk, and loan eligibility with an estimated EMI. Try one of the example requests."
        )
    _trace(trace, "Supervisor Agent", f"Combined {len(responses)} specialist result(s).")
    return AssistantResult(
        answer="\n\n---\n\n".join(responses),
        trace=trace,
        mode=mode,
        mode_detail=mode_detail,
    )
