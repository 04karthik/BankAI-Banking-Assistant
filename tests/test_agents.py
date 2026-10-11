from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bankai import agents, storage
from bankai.agents import (
    calculate_emi,
    fraud_agent,
    loan_agent,
    supervisor_agent,
)
from bankai.llm import LLMSettings


class BankAIAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings_patcher = patch(
            "bankai.agents.load_llm_settings",
            return_value=LLMSettings(
                provider="openai",
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini",
                api_key=None,
            ),
        )
        self.settings_patcher.start()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test-bankai.sqlite3")
        storage.initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.settings_patcher.stop()
        self.temp_dir.cleanup()

    def test_database_is_seeded_and_idempotent(self) -> None:
        storage.initialize_database(self.db_path)
        self.assertEqual(len(storage.list_customers(self.db_path)), 3)
        self.assertEqual(len(storage.get_transactions("CUST-1001", 50, self.db_path)), 8)

    def test_seeded_salary_transactions_match_customer_profiles(self) -> None:
        for customer in storage.CUSTOMERS:
            salary_transactions = [
                transaction
                for transaction in storage.get_transactions(
                    customer["customer_id"], 50, self.db_path
                )
                if transaction["merchant"] == "Monthly salary"
            ]
            self.assertEqual(len(salary_transactions), 1)
            self.assertEqual(salary_transactions[0]["amount"], customer["monthly_income"])

    def test_banking_agent_routes_balance_and_history(self) -> None:
        balance = supervisor_agent(
            "CUST-1001", "What's my balance?", self.db_path
        )
        history = supervisor_agent(
            "CUST-1001", "Show my 3 most recent transactions.", self.db_path
        )
        self.assertIn("$18,420.65", balance.answer)
        self.assertIn("Banking Agent", [step["agent"] for step in balance.trace])
        self.assertEqual(history.answer.count("· **"), 3)
        self.assertNotIn("Fraud Agent", [step["agent"] for step in history.trace])

    def test_banking_agent_can_return_balance_and_history_together(self) -> None:
        result = supervisor_agent(
            "CUST-1001", "Show my balance and 2 recent transactions.", self.db_path
        )
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("2 most recent transactions", result.answer)

    def test_recent_transactions_returns_five_when_available_by_default(self) -> None:
        result = supervisor_agent(
            "CUST-1001", "Show my recent transactions.", self.db_path
        )

        self.assertEqual(result.answer.count("· **"), 5)
        self.assertIn("5 most recent transactions", result.answer)

    def test_fraud_risk_transaction_routes_only_to_fraud_agent(self) -> None:
        result = supervisor_agent(
            "CUST-1001",
            "Check a simulated $950 international transaction for fraud risk",
            self.db_path,
        )

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(
            agent_names,
            ["Supervisor Agent", "Fraud Agent", "Supervisor Agent"],
        )
        self.assertIn("Simulated transaction risk", result.answer)
        self.assertNotIn("most recent transactions", result.answer)

    def test_loan_emi_request_routes_only_to_loan_agent(self) -> None:
        result = supervisor_agent(
            "CUST-1001",
            "Would I qualify for a $25,000 loan over 36 months? Calculate the EMI.",
            self.db_path,
        )

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(
            agent_names,
            ["Supervisor Agent", "Loan Agent", "Supervisor Agent"],
        )
        self.assertIn("Estimated monthly EMI", result.answer)
        self.assertNotIn("Available balance", result.answer)

    def test_ambiguous_transaction_prompt_does_not_run_a_specialist(self) -> None:
        result = supervisor_agent("CUST-1001", "Check this transaction.", self.db_path)

        self.assertEqual(
            [step["agent"] for step in result.trace],
            ["Supervisor Agent", "Supervisor Agent"],
        )
        self.assertNotIn("most recent transactions", result.answer)

    def test_fraud_score_is_deterministic_and_bounded(self) -> None:
        prompt = "Assess a $2,400 online purchase in Singapore at midnight."
        first = fraud_agent(prompt)
        second = fraud_agent(prompt)
        self.assertEqual(first, second)
        self.assertEqual(first[1], 88)
        self.assertEqual(first[2], "High")

    def test_emi_calculation_and_loan_decisions(self) -> None:
        self.assertAlmostEqual(calculate_emi(12000, 0, 12), 1000)
        eligible = loan_agent(
            "CUST-1001", "Check eligibility for a $25,000 loan over 36 months.", self.db_path
        )
        ineligible = loan_agent(
            "CUST-1003", "Check eligibility for a $25,000 loan over 36 months.", self.db_path
        )
        self.assertIn("likely eligible", eligible)
        self.assertIn("not eligible", ineligible)
        self.assertIn("Estimated monthly EMI", eligible)

        zero_loan = loan_agent(
            "CUST-1001", "Check eligibility for a $0 loan over 36 months.", self.db_path
        )
        self.assertIn("valid positive loan amount", zero_loan)

    def test_supervisor_runs_multiple_relevant_agents(self) -> None:
        result = supervisor_agent(
            "CUST-1001",
            "Check my balance, assess risk for a $2,400 online purchase, "
            "and estimate EMI for a $30,000 loan over 48 months.",
            self.db_path,
        )
        agents = [step["agent"] for step in result.trace]
        self.assertEqual(
            agents,
            [
                "Supervisor Agent",
                "Banking Agent",
                "Fraud Agent",
                "Loan Agent",
                "Supervisor Agent",
            ],
        )
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("$2,400.00", result.answer)
        self.assertIn("$30,000.00", result.answer)

    def test_multi_agent_demo_invokes_each_specialist_once(self) -> None:
        prompt = (
            "Check my balance; assess fraud risk for a $2,400 online purchase; "
            "calculate eligibility and EMI for a $30,000 loan over 48 months."
        )
        route_settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=route_settings),
            patch(
                "bankai.agents.classify_question",
                return_value=agents.RoutingPlan(
                    ("banking", "fraud", "loan"), ("balance",)
                ),
            ),
            patch("bankai.agents.banking_agent", wraps=agents.banking_agent) as banking,
            patch("bankai.agents.fraud_agent", wraps=agents.fraud_agent) as fraud,
            patch("bankai.agents.loan_agent", wraps=agents.loan_agent) as loan,
        ):
            result = supervisor_agent("CUST-1001", prompt, self.db_path)

        banking.assert_called_once()
        fraud.assert_called_once()
        loan.assert_called_once()
        self.assertEqual(
            [step["agent"] for step in result.trace],
            [
                "Supervisor Agent",
                "Banking Agent",
                "Fraud Agent",
                "Loan Agent",
                "Supervisor Agent",
            ],
        )
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("Simulated transaction risk", result.answer)
        self.assertIn("Estimated monthly EMI", result.answer)
        self.assertIn("### Banking details", result.answer)
        self.assertIn("### Fraud risk assessment", result.answer)
        self.assertIn("### Loan eligibility and EMI", result.answer)

    def test_unknown_question_does_not_claim_specialist_execution(self) -> None:
        result = supervisor_agent("CUST-1001", "Tell me a fun fact.", self.db_path)
        self.assertEqual(
            [step["agent"] for step in result.trace],
            ["Supervisor Agent", "Supervisor Agent"],
        )
        self.assertIn("account balances", result.answer)


if __name__ == "__main__":
    unittest.main()
