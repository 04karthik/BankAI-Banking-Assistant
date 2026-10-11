from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from bankai.agents import AssistantResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTI_AGENT_PROMPT = (
    "Check my balance, assess risk for a $2,400 online purchase, and estimate EMI "
    "for a $30,000 loan over 48 months."
)


class BankAIStreamlitInteractionTests(unittest.TestCase):
    def test_multi_agent_button_runs_once_and_clear_keeps_sidebar_single(self) -> None:
        customer = {
            "customer_id": "CUST-1001",
            "name": "Alex Morgan",
            "account_id": "BA-4821",
            "balance": 18420.65,
            "currency": "USD",
        }
        result = AssistantResult(
            answer="Consolidated local Banking, Fraud, and Loan results.",
            trace=[
                {"agent": "Supervisor Agent", "status": "Complete", "detail": "Routed"},
                {"agent": "Banking Agent", "status": "Complete", "detail": "Read balance"},
                {"agent": "Fraud Agent", "status": "Complete", "detail": "Assessed risk"},
                {"agent": "Loan Agent", "status": "Complete", "detail": "Calculated EMI"},
                {"agent": "Supervisor Agent", "status": "Complete", "detail": "Combined results"},
            ],
            mode="LIVE LLM MODE",
            mode_detail="Validated route; tools remained local.",
        )

        with (
            patch("bankai.storage.initialize_database"),
            patch("bankai.storage.list_customers", return_value=[customer]),
            patch("bankai.agents.supervisor_agent", return_value=result) as supervisor,
        ):
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(
                sum(element.value == "**Available agents**" for element in app.sidebar.markdown),
                1,
            )
            self.assertEqual(
                len(app.sidebar.success) + len(app.sidebar.info) + len(app.sidebar.warning),
                1,
            )
            app.button(key="demo_Multi-agent").click().run()

            supervisor.assert_called_once_with("CUST-1001", MULTI_AGENT_PROMPT)
            messages = app.session_state["messages"]
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(messages[0]["content"], MULTI_AGENT_PROMPT)
            self.assertEqual(messages[1]["content"], result.answer)
            trace_agents = [step["agent"] for step in messages[1]["trace"]]
            self.assertEqual(
                trace_agents.count("Banking Agent"), 1
            )
            self.assertEqual(trace_agents.count("Fraud Agent"), 1)
            self.assertEqual(trace_agents.count("Loan Agent"), 1)
            self.assertEqual(len(app.expander), 1)
            rendered_trace = [item.value for item in app.markdown]
            for agent in ("Banking Agent", "Fraud Agent", "Loan Agent"):
                self.assertTrue(
                    any(agent in text for text in rendered_trace),
                    f"{agent} was not rendered in the execution trace",
                )
            self.assertEqual(
                sum(element.value == "**Available agents**" for element in app.sidebar.markdown),
                1,
            )
            for agent in ("Supervisor Agent", "Banking Agent", "Fraud Agent", "Loan Agent"):
                self.assertEqual(
                    sum(element.value == f"- {agent}" for element in app.sidebar.markdown),
                    1,
                )

            app.button(key="demo_Multi-agent").click().run()

            supervisor.assert_called_once_with("CUST-1001", MULTI_AGENT_PROMPT)
            self.assertEqual(len(app.session_state["messages"]), 2)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(
                len(app.sidebar.success) + len(app.sidebar.info) + len(app.sidebar.warning),
                1,
            )
            app.sidebar.button[0].click().run()

            self.assertEqual(app.session_state["messages"], [])
            self.assertEqual(supervisor.call_count, 1)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(
                sum(element.value == "**Available agents**" for element in app.sidebar.markdown),
                1,
            )

            app.button(key="demo_Multi-agent").click().run()

            self.assertEqual(supervisor.call_count, 2)
            self.assertEqual(len(app.session_state["messages"]), 2)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(len(app.chat_message), 2)
            self.assertEqual(
                sum(element.value == "**Available agents**" for element in app.sidebar.markdown),
                1,
            )


if __name__ == "__main__":
    unittest.main()