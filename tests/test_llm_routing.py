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
    load_llm_settings,
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

    def test_predefined_multi_agent_prompt_discards_known_non_banking_actions(self) -> None:
        prompt = (
            "Check my balance; assess fraud risk for a $2,400 online purchase; "
            "calculate eligibility and EMI for a $30,000 loan over 48 months."
        )
        result = self._run_live_route(
            prompt,
            {
                "agents": ["banking", "fraud", "loan"],
                "banking_actions": ["balance", "transactions", "emi"],
            },
        )

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertEqual(
            agent_names,
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
        self.assertNotIn("most recent transactions", result.answer)

    def test_predefined_multi_agent_prompt_falls_back_completely_on_invalid_actions(self) -> None:
        prompt = (
            "Check my balance; assess fraud risk for a $2,400 online purchase; "
            "calculate eligibility and EMI for a $30,000 loan over 48 months."
        )
        settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        invalid_response = _provider_response(
            {
                "agents": ["banking", "fraud", "loan"],
                "banking_actions": ["balance", "gift_card_transfer"],
            }
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=settings),
            patch("bankai.llm.urlopen", side_effect=[invalid_response, invalid_response]),
        ):
            result = supervisor_agent("CUST-1001", prompt, self.db_path)

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("invalid banking actions", result.mode_detail)
        self.assertEqual(
            agent_names,
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

    def test_live_route_balance_query(self) -> None:
        result = self._run_live_route(
            "What's my current account balance?",
            {"agents": ["banking"], "banking_actions": ["balance"]},
        )
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("Banking Agent", [step["agent"] for step in result.trace])
        self.assertIn("$18,420.65", result.answer)

    def test_live_route_transactions_query(self) -> None:
        result = self._run_live_route(
            "Show my 5 most recent transactions.",
            {"agents": ["banking"], "banking_actions": ["transactions"]},
        )
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("Banking Agent", [step["agent"] for step in result.trace])
        self.assertEqual(result.answer.count("· **"), 5)

    def test_live_route_last_transaction_query(self) -> None:
        result = self._run_live_route(
            "last transaction",
            {"agents": ["banking"], "banking_actions": ["transactions"]},
        )
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("Banking Agent", [step["agent"] for step in result.trace])
        self.assertIn("Here are the", result.answer)

    def test_live_route_fraud_risk_query(self) -> None:
        result = self._run_live_route(
            "Check a simulated $950 international transaction for fraud risk",
            {
                "agents": ["banking", "fraud"],
                "banking_actions": ["transactions"],
            },
        )
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("Fraud Agent", [step["agent"] for step in result.trace])
        self.assertNotIn("Banking Agent", [step["agent"] for step in result.trace])
        self.assertIn("Simulated transaction risk", result.answer)
        self.assertNotIn("most recent transactions", result.answer)

    def test_live_route_loan_and_emi_query(self) -> None:
        result = self._run_live_route(
            "Would I qualify for a $25,000 loan over 36 months? Calculate the EMI.",
            {"agents": ["loan"], "banking_actions": ["emi", "interest_rate"]},
        )
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("Loan Agent", [step["agent"] for step in result.trace])
        self.assertNotIn("Banking Agent", [step["agent"] for step in result.trace])
        self.assertIn("Estimated monthly EMI", result.answer)
        self.assertNotIn("Available balance", result.answer)

    def test_ambiguous_transaction_request_ignores_model_agent_selection(self) -> None:
        result = self._run_live_route(
            "Check this transaction.",
            {"agents": ["banking"], "banking_actions": ["transactions"]},
        )

        self.assertEqual(
            [step["agent"] for step in result.trace],
            ["Supervisor Agent", "Supervisor Agent"],
        )
        self.assertNotIn("most recent transactions", result.answer)

    def test_invalid_model_output_falls_back_to_fraud_only_intent(self) -> None:
        malformed_response = _provider_response(
            {
                "agents": ["banking", "fraud"],
                "banking_actions": ["wire_transfer"],
            }
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=self.settings),
            patch(
                "bankai.llm.urlopen",
                side_effect=[malformed_response, malformed_response],
            ),
        ):
            result = supervisor_agent(
                "CUST-1001",
                "Check a simulated $950 international transaction for fraud risk",
                self.db_path,
            )

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("Fraud Agent", agent_names)
        self.assertNotIn("Banking Agent", agent_names)
        self.assertIn("Simulated transaction risk", result.answer)
        self.assertNotIn("most recent transactions", result.answer)

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

    def test_ollama_configuration_is_keyless_and_local_only(self) -> None:
        with patch.dict(
            "os.environ",
            {"BANKAI_LLM_PROVIDER": "ollama"},
            clear=True,
        ):
            local_settings = load_llm_settings()

        self.assertEqual(local_settings.base_url, "http://localhost:11434/v1")
        self.assertEqual(local_settings.model, "llama3.2")
        self.assertIsNone(local_settings.api_key)
        self.assertTrue(local_settings.ready)
        self.assertEqual(local_settings.timeout_seconds, 60.0)

        with patch.dict(
            "os.environ",
            {
                "BANKAI_LLM_PROVIDER": "ollama",
                "BANKAI_LLM_BASE_URL": "https://ollama.example.test/v1",
            },
            clear=True,
        ):
            remote_settings = load_llm_settings()

        self.assertFalse(remote_settings.ready)
        self.assertIn("localhost endpoint", remote_settings.config_error)

    def test_no_cloud_key_defaults_to_ollama_and_openai_key_keeps_remote_provider(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            local_settings = load_llm_settings()

        self.assertEqual(local_settings.provider, "ollama")
        self.assertEqual(local_settings.base_url, "http://localhost:11434/v1")
        self.assertEqual(local_settings.model, "llama3.2")
        self.assertIsNone(local_settings.api_key)
        self.assertTrue(local_settings.ready)

        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": "openai-test-key"}, clear=True
        ):
            remote_settings = load_llm_settings()

        self.assertEqual(remote_settings.provider, "openai")
        self.assertEqual(remote_settings.api_key, "openai-test-key")
        self.assertTrue(remote_settings.ready)

    def test_ollama_request_omits_authentication_header(self) -> None:
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            settings = load_llm_settings()

        with patch(
            "bankai.llm.urlopen",
            return_value=_provider_response(
                {"agents": ["banking"], "banking_actions": ["balance"]}
            ),
        ) as open_url:
            plan = classify_question("Check my balance.", settings)

        request = open_url.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertEqual(plan.agents, ("banking",))
        self.assertEqual(request.full_url, "http://localhost:11434/v1/chat/completions")
        self.assertEqual(request_body["model"], "llama3.2")
        self.assertIsNone(request.get_header("Authorization"))

    def test_ollama_unavailable_falls_back_to_mock_router(self) -> None:
        settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=settings),
            patch("bankai.llm.urlopen", side_effect=TimeoutError),
        ):
            result = supervisor_agent("CUST-1001", "What's my balance?", self.db_path)

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("timed out", result.mode_detail)

    def test_ollama_invalid_output_falls_back_to_mock_router(self) -> None:
        settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        invalid_response = _provider_response(
            {"agents": ["banking", "admin"], "banking_actions": ["balance"]}
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=settings),
            patch("bankai.llm.urlopen", side_effect=[invalid_response, invalid_response]),
        ):
            result = supervisor_agent("CUST-1001", "What's my balance?", self.db_path)

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("$18,420.65", result.answer)
        self.assertIn("invalid agent selections", result.mode_detail)

    def test_ollama_explicit_intents_override_incomplete_valid_route(self) -> None:
        settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        prompt = (
            "Check my balance, assess risk for a $2,400 online purchase, "
            "and estimate EMI for a $30,000 loan over 48 months."
        )
        partial_response = _provider_response(
            {"agents": ["banking", "fraud"], "banking_actions": ["balance"]}
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=settings),
            patch("bankai.llm.urlopen", return_value=partial_response),
        ):
            result = supervisor_agent("CUST-1001", prompt, self.db_path)

        agent_names = [step["agent"] for step in result.trace]
        self.assertEqual(result.mode, "LIVE LLM MODE")
        self.assertIn("explicit user intents determined agent routing", result.mode_detail)
        self.assertIn("Banking Agent", agent_names)
        self.assertIn("Fraud Agent", agent_names)
        self.assertIn("Loan Agent", agent_names)
        self.assertIn("Estimated monthly EMI", result.answer)

    def test_ollama_loan_and_fraud_routes_ignore_irrelevant_banking_actions(self) -> None:
        settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        cases = (
            (
                "Would I qualify for a $25,000 loan over 36 months? Calculate the EMI.",
                {
                    "agents": ["banking", "loan"],
                    "banking_actions": ["balance", "transactions"],
                },
                "Loan Agent",
                "Estimated monthly EMI",
            ),
            (
                "Is a $950 online purchase in Singapore suspicious? Assess fraud risk.",
                {"agents": ["fraud"], "banking_actions": ["transactions"]},
                "Fraud Agent",
                "Simulated transaction risk",
            ),
        )
        for prompt, route, expected_agent, expected_answer in cases:
            with self.subTest(expected_agent=expected_agent):
                with (
                    patch("bankai.agents.load_llm_settings", return_value=settings),
                    patch("bankai.llm.urlopen", return_value=_provider_response(route)),
                ):
                    result = supervisor_agent("CUST-1001", prompt, self.db_path)

                agent_names = [step["agent"] for step in result.trace]
                self.assertEqual(result.mode, "LIVE LLM MODE")
                self.assertIn(expected_agent, agent_names)
                self.assertNotIn("Banking Agent", agent_names)
                self.assertIn(expected_answer, result.answer)

    def test_ollama_invalid_banking_action_falls_back_safely(self) -> None:
        settings = LLMSettings(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            model="llama3.2",
            api_key=None,
        )
        malformed_response = _provider_response(
            {
                "agents": ["banking", "fraud"],
                "banking_actions": ["wire_transfer"],
            }
        )
        with (
            patch("bankai.agents.load_llm_settings", return_value=settings),
            patch(
                "bankai.llm.urlopen",
                side_effect=[malformed_response, malformed_response],
            ),
        ):
            result = supervisor_agent(
                "CUST-1001",
                "Is a $950 online purchase in Singapore suspicious?",
                self.db_path,
            )

        self.assertEqual(result.mode, "MOCK MODE")
        self.assertIn("invalid banking actions", result.mode_detail)
        self.assertIn("Fraud Agent", [step["agent"] for step in result.trace])
        self.assertIn("Simulated transaction risk", result.answer)

    def test_non_banking_routes_discard_irrelevant_action_metadata(self) -> None:
        routes = (
            {"agents": ["loan"], "banking_actions": ["calculate_emi", "interest_rate"]},
            {"agents": ["fraud"], "banking_actions": ["risk_assessment"]},
        )
        for route in routes:
            with self.subTest(route=route):
                plan = validate_route_output(route)
                self.assertEqual(plan.banking_actions, ())

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
            {"agents": ["banking"], "banking_actions": ["wire_transfer"]},
            {"agents": ["loan"], "banking_actions": ["wire_transfer"]},
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

    def test_groq_provider_defaults_and_env_autodetect(self) -> None:
        with patch.dict(
            "os.environ",
            {"GROQ_API_KEY": "gsk_test12345"},
            clear=True,
        ):
            from bankai.llm import load_llm_settings
            settings = load_llm_settings()
            self.assertEqual(settings.provider, "groq")
            self.assertEqual(settings.base_url, "https://api.groq.com/openai/v1")
            self.assertEqual(settings.model, "openai/gpt-oss-120b")
            self.assertEqual(settings.api_key, "gsk_test12345")
            self.assertTrue(settings.ready)
            self.assertIsNone(settings.config_error)


    def test_openrouter_provider_defaults_and_env_autodetect(self) -> None:
        with patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "sk-or-v1-test12345"},
            clear=True,
        ):
            from bankai.llm import load_llm_settings
            settings = load_llm_settings()
            self.assertEqual(settings.provider, "openrouter")
            self.assertEqual(settings.base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(settings.model, "openrouter/free")
            self.assertEqual(settings.api_key, "sk-or-v1-test12345")
            self.assertTrue(settings.ready)
            self.assertIsNone(settings.config_error)

    def test_markdown_fenced_json_parsing(self) -> None:
        raw_fenced = "```json\n{\"agents\": [\"banking\"], \"banking_actions\": [\"balance\"]}\n```"
        envelope = {"choices": [{"message": {"content": raw_fenced}}]}
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            plan = classify_question("Check balance", self.settings)
            self.assertEqual(plan.agents, ("banking",))
            self.assertEqual(plan.banking_actions, ("balance",))


    def test_openrouter_error_envelope_extracted_safely(self) -> None:
        err_envelope = {"error": {"message": "OpenRouter free quota reached"}}
        settings = LLMSettings(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/free",
            api_key="sk-or-v1-test",
        )
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(err_envelope).encode("utf-8"))):
            with self.assertRaisesRegex(LLMError, "OpenRouter error: OpenRouter free quota reached"):
                classify_question("Check balance", settings)
    def test_loan_and_emi_routing_json_parsing(self) -> None:
        raw_loan_response = "Here is the route: ```json\n{\"agents\": [\"loan\"], \"banking_actions\": []}\n```"
        envelope = {"choices": [{"message": {"content": raw_loan_response}}]}
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            plan = classify_question("Calculate my loan EMI for $25,000 over 36 months", self.settings)
            self.assertEqual(plan.agents, ("loan",))
            self.assertEqual(plan.banking_actions, ())


    def test_extract_text_content_from_strings_and_lists(self) -> None:
        from bankai.llm import _extract_text_content

        # Case 1: Pure string
        self.assertEqual(_extract_text_content("hello world"), "hello world")

        # Case 2: List of text blocks with type: text
        list_content = [{"type": "text", "text": "{\"agents\":[\"loan\"],\"banking_actions\":[]}"}]
        self.assertEqual(_extract_text_content(list_content), "{\"agents\":[\"loan\"],\"banking_actions\":[]}")

        # Case 3: List of dicts with text key
        list_content_simple = [{"text": "part1 "}, {"text": "part2"}]
        self.assertEqual(_extract_text_content(list_content_simple), "part1 part2")

        # Case 4: List of mixed strings and dicts
        mixed_list = ["string1 ", {"text": "string2"}]
        self.assertEqual(_extract_text_content(mixed_list), "string1 string2")

    def test_invalid_content_raises_llmerror(self) -> None:
        from bankai.llm import _extract_text_content

        invalid_contents = [
            12345,
            True,
            None,
            [{"type": "image_url", "image_url": "https://example.test/img.png"}],
            [],
        ]
        for item in invalid_contents:
            with self.subTest(item=item), self.assertRaises(LLMError):
                _extract_text_content(item)


    def test_normalize_json_text_regression_suite(self) -> None:
        from bankai.llm import _normalize_json_text

        # 1. Valid raw JSON
        self.assertEqual(_normalize_json_text('{"agents": ["banking"], "banking_actions": ["balance"]}'), '{"agents": ["banking"], "banking_actions": ["balance"]}')

        # 2. Markdown fenced JSON
        fenced = "```json\n{\n  \"agents\": [\"loan\"],\n  \"banking_actions\": []\n}\n```"
        self.assertEqual(_normalize_json_text(fenced), '{\n  "agents": ["loan"],\n  "banking_actions": []\n}')

        # 3. Extra surrounding whitespace & commentary
        commentary = "   Here is the JSON output:\n```json\n{\"agents\": [\"fraud\"], \"banking_actions\": []}\n```\nHope this helps!   "
        self.assertEqual(_normalize_json_text(commentary), '{"agents": ["fraud"], "banking_actions": []}')

    def test_invalid_json_diagnostics_preserves_mock_fallback(self) -> None:
        invalid_json_response = "I am an AI assistant and cannot produce JSON right now."
        envelope = {"choices": [{"message": {"content": invalid_json_response}}]}
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            with self.assertRaises(LLMError) as ctx:
                classify_question("Check balance", self.settings)
            self.assertIn("The LLM response was not valid JSON", str(ctx.exception))
            self.assertIn("I am an AI assistant", str(ctx.exception))


    def test_extract_text_content_from_reasoning_message_envelope(self) -> None:
        from bankai.llm import _extract_text_content

        # Case 5: Message object with null content but reasoning_content field
        message_obj = {"content": None, "reasoning_content": "{\"agents\":[\"fraud\"],\"banking_actions\":[]}"}
        self.assertEqual(_extract_text_content(None, message_obj), "{\"agents\":[\"fraud\"],\"banking_actions\":[]}")

    def test_safety_preamble_stripped_from_openrouter_content(self) -> None:
        raw_response = "User Safety: safe\n{\"agents\": [\"banking\"], \"banking_actions\": [\"transactions\"]}"
        envelope = {"choices": [{"message": {"content": raw_response}}]}
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            plan = classify_question("last transaction", self.settings)
            self.assertEqual(plan.agents, ("banking",))
            self.assertEqual(plan.banking_actions, ("transactions",))

    def test_safety_only_content_falls_back_to_reasoning_if_available(self) -> None:
        envelope = {
            "choices": [
                {
                    "message": {
                        "content": "User Safety: safe",
                        "reasoning_content": "{\"agents\": [\"banking\"], \"banking_actions\": [\"transactions\"]}",
                    }
                }
            ]
        }
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            plan = classify_question("last transaction", self.settings)
            self.assertEqual(plan.agents, ("banking",))
            self.assertEqual(plan.banking_actions, ("transactions",))

    def test_pure_safety_metadata_without_json_falls_back_to_mock_router(self) -> None:
        # When model returns ONLY safety string and no JSON, fallback triggers cleanly
        envelope = {"choices": [{"message": {"content": "User Safety: safe"}}]}
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            with self.assertRaises(LLMError) as ctx:
                classify_question("last transaction", self.settings)
            self.assertIn("The LLM response was not valid JSON", str(ctx.exception))

    def test_invalid_json_retries_once_and_succeeds(self) -> None:
        invalid_envelope = {"choices": [{"message": {"content": "Not JSON text"}}]}
        valid_envelope = {
            "choices": [
                {
                    "message": {
                        "content": '{"agents": ["banking", "loan"], "banking_actions": ["balance"]}'
                    }
                }
            ]
        }
        responses = [
            io.BytesIO(json.dumps(invalid_envelope).encode("utf-8")),
            io.BytesIO(json.dumps(valid_envelope).encode("utf-8")),
        ]
        with patch("bankai.llm.urlopen", side_effect=responses):
            plan = classify_question("Check balance and loan eligibility", self.settings)
            self.assertEqual(plan.agents, ("banking", "loan"))
            self.assertEqual(plan.banking_actions, ("balance",))

    def test_structured_nested_content_blocks_extracted(self) -> None:
        nested_content = [
            {
                "type": "text",
                "text": '{"agents": ["banking", "fraud"], "banking_actions": ["balance"]}',
            }
        ]
        envelope = {"choices": [{"message": {"content": nested_content}}]}
        with patch("bankai.llm.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode("utf-8"))):
            plan = classify_question("Check balance and fraud risk", self.settings)
            self.assertEqual(plan.agents, ("banking", "fraud"))
            self.assertEqual(plan.banking_actions, ("balance",))


if __name__ == "__main__":
    unittest.main()
