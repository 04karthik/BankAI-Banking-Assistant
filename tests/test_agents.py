from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bankai import storage
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

    def test_supervisor_runs_multiple_relevant_agents(self) -> None:
        result = supervisor_agent(
            "CUST-1001",
            "Check my balance, assess risk for a $2,400 online purchase, "
            "and estimate EMI for a $30,000 loan over 48 months.",
            self.db_path,
        )
        agents = [step["agent"] for step in result.trace]
        self.assertIn("Banking Agent", agents)
        self.assertIn("Fraud Agent", agents)
        self.assertIn("Loan Agent", agents)
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("$2,400.00", result.answer)
        self.assertIn("$30,000.00", result.answer)

    def test_unknown_question_does_not_claim_specialist_execution(self) -> None:
        result = supervisor_agent("CUST-1001", "Tell me a fun fact.", self.db_path)
        self.assertEqual(
            [step["agent"] for step in result.trace],
            ["Supervisor Agent", "Supervisor Agent"],
        )
        self.assertIn("account balances", result.answer)


if __name__ == "__main__":
    unittest.main()
