from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from bankai import storage
from bankai.agents import supervisor_agent
from bankai.llm import (
    LLMError,
    LLMSettings,
    classify_question,
    validate_route_output,
)


def _provider_response(route: dict[str, object]) -> io.BytesIO:
    envelope = {"choices": [{"message": {"content": json.dumps(route)}}]}
    return io.BytesIO(json.dumps(envelope).encode("utf-8"))


def _provider_http_error(status_code: int) -> HTTPError:
    return HTTPError(
        "https://llm.example.test/v1/chat/completions",
        status_code,
        "Provider error",
        hdrs=None,
        fp=io.BytesIO(b"provider details"),
    )


class LLMRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test-bankai.sqlite3")
        storage.initialize_database(self.db_path)
        self.settings = LLMSettings(
            provider="custom",
            base_url="https://llm.example.test/v1",
            model="test-router",
            api_key="test-only-key",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_live_route(
        self,
        question: str,
        route: dict[str, object],
    ):
        with (
            patch("bankai.agents.load_llm_settings", return_value=self.settings),
            patch("bankai.llm.urlopen", return_value=_provider_response(route)),
        ):
            return supervisor_agent("CUST-1001", question, self.db_path)

    def test_validated_intent_routes_indirect_transaction_question(self) -> None:
        result = self._run_live_route(
            "What payments have come out of my account recently?",
            {"agents": ["banking"], "banking_actions": ["transactions"]},
        )

        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertEqual(result.answer.count("· **"), 5)
        self.assertNotIn("Available balance", result.answer)
        self.assertIn("Banking Agent", [step["agent"] for step in result.trace])
        self.assertIn("LIVE LLM MODE", result.trace[0]["detail"])

    def test_live_multi_agent_route_keeps_tools_deterministic(self) -> None:
        result = self._run_live_route(
            "Check my balance, assess risk for a $2,400 online purchase in Singapore, "
            "and estimate EMI for a $30,000 loan over 48 months.",
            {
                "agents": ["banking", "fraud", "loan"],
                "banking_actions": ["balance"],
            },
        )

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("Banking Agent", agent_names)
        self.assertIn("Fraud Agent", agent_names)
        self.assertIn("Loan Agent", agent_names)
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("$2,400.00", result.answer)
        self.assertIn("$30,000.00", result.answer)
        self.assertIn("78/100", result.answer)
        self.assertIn("Estimated monthly EMI", result.answer)

    def test_no_api_key_uses_existing_mock_router(self) -> None:
        settings = LLMSettings(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
            api_key=None,
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=settings),
            patch("bankai.llm.urlopen") as open_url,
        ):
            result = supervisor_agent("CUST-1001", "What's my balance?", self.db_path)

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("No API key configured", result.mode_detail)
        open_url.assert_not_called()

    def test_invalid_structured_output_falls_back_to_mock_router(self) -> None:
        result = self._run_live_route(
            "What's my balance?",
            {"agents": ["banking", "admin"], "banking_actions": ["balance"]},
        )

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("invalid agent selections", result.mode_detail)

    def test_timeout_falls_back_to_mock_router(self) -> None:
        with (
            patch("bankai.agents.load_llm_settings", return_value=self.settings),
            patch("bankai.llm.urlopen", side_effect=TimeoutError),
        ):
            result = supervisor_agent("CUST-1001", "What's my balance?", self.db_path)

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("timed out", result.mode_detail)

    def test_http_error_falls_back_without_exposing_provider_body(self) -> None:
        with (
            patch("bankai.agents.load_llm_settings", return_value=self.settings),
            patch("bankai.llm.urlopen", side_effect=_provider_http_error(503)),
        ):
            result = supervisor_agent("CUST-1001", "What's my balance?", self.db_path)

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("HTTP 503", result.mode_detail)
        self.assertNotIn("provider details", result.mode_detail)

    def test_auth_model_and_rate_limit_errors_fall_back_safely(self) -> None:
        cases = (
            (401, "API-key authentication"),
            (404, "configured model or endpoint"),
            (429, "rate limit or quota"),
        )
        for status_code, expected_message in cases:
            with self.subTest(status_code=status_code):
                with (
                    patch("bankai.agents.load_llm_settings", return_value=self.settings),
                    patch(
                        "bankai.llm.urlopen",
                        side_effect=_provider_http_error(status_code),
                    ),
                ):
                    result = supervisor_agent(
                        "CUST-1001", "What's my balance?", self.db_path
                    )

                self.assertEqual(result.mode, "MOCK MODE")
                self.assertIn(expected_message, result.mode_detail)
                self.assertNotIn("test-only-key", result.mode_detail)
                self.assertNotIn("provider details", result.mode_detail)

    def test_route_validator_rejects_extra_fields_and_inconsistent_actions(self) -> None:
        invalid_plans = (
            {"agents": ["banking"], "banking_actions": ["balance"], "amount": 100},
            {"agents": ["banking"], "banking_actions": []},
            {"agents": [], "banking_actions": ["transactions"]},
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan), self.assertRaises(LLMError):
                validate_route_output(plan)

    def test_gemini_request_uses_supported_reasoning_budget(self) -> None:
        settings = LLMSettings(
            provider="custom",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-3.8-flash",
            api_key="test-only-key",
        )
        with patch(
            "bankai.llm.urlopen",
            return_value=_provider_response(
                {"agents": ["banking"], "banking_actions": ["balance"]}
            ),
        ) as open_url:
            plan = classify_question("Check the balance.", settings)

        request = open_url.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertEqual(plan.agents, ("banking",))
        self.assertEqual(request.full_url, f"{settings.base_url}/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-key")
        self.assertEqual(request_body["reasoning_effort"], "low")
        self.assertEqual(request_body["max_tokens"], 1024)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
