"""Optional OpenAI-compatible intent routing with strict local validation."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openrouter": "openrouter/free",
    "groq": "openai/gpt-oss-120b",
    "ollama": "llama3.2",
}
_ALLOWED_AGENTS = frozenset({"banking", "fraud", "loan"})
_ALLOWED_BANKING_ACTIONS = frozenset({"balance", "transactions"})
_NON_BANKING_ACTION_LABELS = frozenset(
    {"emi", "calculate_emi", "interest_rate", "loan_eligibility", "fraud_risk", "risk_assessment"}
)
_MAX_RESPONSE_BYTES = 1_000_000
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_SYSTEM_PROMPT = """You are an intent routing classifier for a fictional banking demo.
Do NOT include safety tags, safety preamble text (such as "User Safety: safe"), or markdown formatting.
Return EXACTLY ONE raw JSON object with this shape:

{"agents": [], "banking_actions": []}

Rules for `agents` (list containing 1 to 3 valid agent IDs):
- Include "banking" for questions asking about account balances, checking details, or to view transaction history.
- Mentioning a transaction or purchase as the subject of a fraud/risk question does NOT request transaction history; select "fraud" only unless the user separately asks to view account data.
- Include "fraud" for questions asking about transaction risk, fraud, or suspicious purchases.
- Include "loan" for questions asking about loans, borrowing eligibility, EMI, monthly installments, or interest rates.
- Include multiple agent IDs if the question asks for multiple distinct banking tasks.

Rules for `banking_actions` (list containing 0 to 2 action IDs):
- `banking_actions` describes local Banking Agent actions only; it does not select agents.
- Never place loan calculations, eligibility, interest rates, or fraud/risk labels in `banking_actions`.
- If "banking" is in `agents`, include at least one of ["balance", "transactions"].
  - Include "transactions" if the question asks for transaction history, recent activity, last transaction, or statements.
  - Include "balance" if the question asks for account balance or checking details.
- If "banking" is NOT in `agents`, `banking_actions` MUST be exactly []. This applies to loan-only and fraud-only requests.

Examples:
- Loan eligibility or EMI only: {"agents": ["loan"], "banking_actions": []}
- Fraud or purchase-risk only: {"agents": ["fraud"], "banking_actions": []}
- Account balance only: {"agents": ["banking"], "banking_actions": ["balance"]}
- Loan and fraud together, with no account request: {"agents": ["loan", "fraud"], "banking_actions": []}
- "Check a simulated $950 international transaction for fraud risk": {"agents": ["fraud"], "banking_actions": []}
- "Check this transaction" without more detail: {"agents": [], "banking_actions": []}

For completely unrelated non-banking questions, return:
{"agents": [], "banking_actions": []}

OUTPUT FORMAT: Return raw JSON only. No safety commentary, no preamble text, no backticks."""


class LLMError(RuntimeError):
    """A safe-to-display LLM configuration, request, or output error."""


@dataclass(frozen=True)
class RoutingPlan:
    agents: tuple[str, ...]
    banking_actions: tuple[str, ...]


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    model: str
    api_key: str | None = field(repr=False)
    timeout_seconds: float = 10.0
    config_error: str | None = None

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def ready(self) -> bool:
        if self.config_error is not None:
            return False
        if self.provider == "ollama":
            return _is_local_ollama_endpoint(self.base_url)
        return self.has_api_key


def _is_local_ollama_endpoint(base_url: str) -> bool:
    parsed = urlsplit(base_url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in _LOCAL_HOSTS


def _validate_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    is_local_http = parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS
    if (
        not parsed.hostname
        or (parsed.scheme != "https" and not is_local_http)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "The LLM base URL must use HTTPS (HTTP is allowed only for localhost)."
        )
    return normalized


def load_llm_settings() -> LLMSettings:
    """Load provider configuration; the API key is read only from the environment."""
    raw_provider = os.environ.get("BANKAI_LLM_PROVIDER", "").strip().lower()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    bankai_key = os.environ.get("BANKAI_LLM_API_KEY", "").strip()

    if raw_provider:
        provider = raw_provider
    elif openrouter_key:
        provider = "openrouter"
    elif groq_key:
        provider = "groq"
    elif openai_key or bankai_key:
        provider = "openai"
    else:
        provider = "ollama"

    api_key = None if provider == "ollama" else (
        bankai_key
        or (openrouter_key if provider == "openrouter" else "")
        or (groq_key if provider == "groq" else "")
        or openai_key
        or None
    )

    base_url = (
        os.environ.get("BANKAI_LLM_BASE_URL", "").strip()
        or _DEFAULT_BASE_URLS.get(provider, "")
    )
    model = (
        os.environ.get("BANKAI_LLM_MODEL", "").strip()
        or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    )
    default_timeout = "60" if provider == "ollama" else "10"
    timeout_raw = os.environ.get(
        "BANKAI_LLM_TIMEOUT_SECONDS", default_timeout
    ).strip()
    timeout_seconds = float(default_timeout)
    config_error: str | None = None

    if api_key or provider == "ollama":
        if provider not in {"openai", "openrouter", "groq", "custom", "ollama"}:
            config_error = (
                "BANKAI_LLM_PROVIDER must be openai, openrouter, groq, custom, or ollama."
            )
        elif not base_url:
            config_error = "BANKAI_LLM_BASE_URL is required for the custom provider."
        elif not model or len(model) > 128:
            config_error = "BANKAI_LLM_MODEL must contain between 1 and 128 characters."
        else:
            try:
                timeout_seconds = float(timeout_raw)
                if not 1 <= timeout_seconds <= 60:
                    raise ValueError
            except ValueError:
                timeout_seconds = 10.0
                config_error = "BANKAI_LLM_TIMEOUT_SECONDS must be between 1 and 60."
            if config_error is None:
                try:
                    base_url = _validate_base_url(base_url)
                except ValueError as error:
                    config_error = str(error)
            if (
                config_error is None
                and provider == "ollama"
                and not _is_local_ollama_endpoint(base_url)
            ):
                config_error = "Ollama must use a localhost endpoint; remote URLs are not allowed."

    return LLMSettings(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        config_error=config_error,
    )


def validate_route_output(output: object) -> RoutingPlan:
    """Validate an untrusted route; only Banking routes may carry executable actions."""
    if not isinstance(output, dict) or set(output) != {"agents", "banking_actions"}:
        raise LLMError("The LLM response did not match the required routing schema.")

    agents = output["agents"]
    actions = output["banking_actions"]
    if (
        not isinstance(agents, list)
        or len(agents) > len(_ALLOWED_AGENTS)
        or any(type(agent) is not str or agent not in _ALLOWED_AGENTS for agent in agents)
        or len(set(agents)) != len(agents)
    ):
        raise LLMError("The LLM response contained invalid agent selections.")
    if (
        not isinstance(actions, list)
        or len(actions) > len(_ALLOWED_BANKING_ACTIONS) + len(_NON_BANKING_ACTION_LABELS)
        or any(type(action) is not str for action in actions)
        or len(set(actions)) != len(actions)
    ):
        raise LLMError("The LLM response contained invalid banking actions.")

    if any(
        action not in _ALLOWED_BANKING_ACTIONS
        and action not in _NON_BANKING_ACTION_LABELS
        for action in actions
    ):
        raise LLMError("The LLM response contained invalid banking actions.")
    actions = [action for action in actions if action in _ALLOWED_BANKING_ACTIONS]
    if len(actions) > len(_ALLOWED_BANKING_ACTIONS):
        raise LLMError("The LLM response contained invalid banking actions.")

    has_banking_agent = "banking" in agents
    if has_banking_agent:
        if not actions:
            raise LLMError("The LLM response had inconsistent banking agent actions.")
    elif not agents and actions:
        raise LLMError("The LLM response had inconsistent banking agent actions.")
    else:
        # These labels cannot select or invoke tools without the Banking Agent.
        # Discard them rather than failing valid Fraud/Loan classifications.
        actions = []

    return RoutingPlan(tuple(agents), tuple(actions))


def _provider_name(base_url: str) -> str:
    hostname = urlsplit(base_url).hostname
    if hostname == "generativelanguage.googleapis.com":
        return "Gemini"
    if hostname == "api.groq.com":
        return "Groq"
    if hostname == "openrouter.ai":
        return "OpenRouter"
    return "LLM provider"


def _strip_safety_preambles(text: str) -> str:
    """Remove safety preamble lines such as 'User Safety: safe' or '[Safety] safe'."""
    if not text:
        return text
    import re
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(
            r"^(?:user\s+|system\s+)?safety(?:\s+evaluation|\s+status|\s+rating)?\s*:\s*\w+$",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue
        if re.match(r"^\[safety\]\s*\w+$", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _extract_block_text(block: object) -> str:
    """Extract string content from structured text blocks or nested content objects."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        for key in ("text", "content", "value"):
            val = block.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list):
                extracted = "".join(_extract_block_text(item) for item in val)
                if extracted.strip():
                    return extracted
            if isinstance(val, dict):
                extracted = _extract_block_text(val)
                if extracted.strip():
                    return extracted
    return ""


def _extract_text_content(content: object, message_obj: dict[str, Any] | None = None) -> str:
    """Extract string text content from a string, list, dict, or message envelope object."""
    candidates: list[str] = []

    if isinstance(content, str):
        if content.strip():
            candidates.append(content)
    elif isinstance(content, list):
        parts = [_extract_block_text(block) for block in content]
        joined = "".join(p for p in parts if p.strip())
        if joined:
            candidates.append(joined)
    elif isinstance(content, dict):
        extracted = _extract_block_text(content)
        if extracted.strip():
            candidates.append(extracted)

    if isinstance(message_obj, dict):
        for key in ("reasoning", "reasoning_content", "text"):
            val = message_obj.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(val)
            elif isinstance(val, (list, dict)):
                extracted = _extract_block_text(val)
                if extracted.strip():
                    candidates.append(extracted)

    for candidate in candidates:
        cleaned = _strip_safety_preambles(candidate)
        if "{" in cleaned and "}" in cleaned:
            return cleaned

    for candidate in candidates:
        cleaned = _strip_safety_preambles(candidate)
        if cleaned:
            return cleaned

    for candidate in candidates:
        if candidate.strip():
            return candidate.strip()

    raise LLMError("The LLM provider returned non-string message content.")


def _request_completion(
    message: str,
    settings: LLMSettings,
    messages_override: list[dict[str, Any]] | None = None,
) -> str:
    if not settings.ready or (settings.provider != "ollama" and not settings.api_key):
        raise LLMError(settings.config_error or "Live LLM routing is not configured.")

    hostname = urlsplit(settings.base_url).hostname
    messages = messages_override or [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    request_body: dict[str, Any] = {
        "model": settings.model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": messages,
    }
    if hostname != "openrouter.ai":
        request_body["response_format"] = {"type": "json_object"}

    if hostname == "generativelanguage.googleapis.com":
        request_body["max_tokens"] = 1024
        request_body["reasoning_effort"] = "low"

    headers = {"Content-Type": "application/json"}
    if settings.provider != "ollama":
        headers["Authorization"] = f"Bearer {settings.api_key}"
    if hostname == "openrouter.ai":
        headers["HTTP-Referer"] = "http://localhost:5000"
        headers["X-Title"] = "BankAI Banking Assistant"

    request = Request(
        f"{settings.base_url}/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=settings.timeout_seconds) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        try:
            err_body = error.read(_MAX_RESPONSE_BYTES).decode("utf-8")
            err_json = json.loads(err_body)
            if isinstance(err_json, dict) and "error" in err_json:
                detail = err_json["error"]
                msg = detail.get("message") if isinstance(detail, dict) else str(detail)
                if msg:
                    raise LLMError(f"{_provider_name(settings.base_url)} (HTTP {error.code}): {msg}")
        except Exception:
            pass
        raise LLMError(_http_error_message(error.code, settings.base_url)) from None
    except (TimeoutError, socket.timeout):
        raise LLMError("The LLM request timed out.") from None
    except URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise LLMError("The LLM request timed out.") from None
        raise LLMError("The LLM provider could not be reached.") from None
    except OSError:
        raise LLMError("The LLM provider could not be reached.") from None

    if len(body) > _MAX_RESPONSE_BYTES:
        raise LLMError("The LLM provider response was too large.")

    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LLMError("The LLM provider returned invalid JSON response bytes.") from None

    if isinstance(envelope, dict) and "error" in envelope:
        detail = envelope["error"]
        msg = detail.get("message") if isinstance(detail, dict) else str(detail)
        raise LLMError(f"{_provider_name(settings.base_url)} error: {msg}")

    try:
        message_obj = envelope["choices"][0]["message"]
        content = message_obj.get("content")
    except (KeyError, IndexError, TypeError):
        raise LLMError("The LLM provider response did not contain message content.") from None

    return _extract_text_content(content, message_obj if isinstance(message_obj, dict) else None)


def _http_error_message(status_code: int, base_url: str) -> str:
    """Return actionable provider errors without echoing response bodies or keys."""
    provider = _provider_name(base_url)
    if status_code == 401:
        return (
            f"{provider} rejected API-key authentication (HTTP 401). "
            "Verify your API key and restart the workflow after rotating it."
        )
    if status_code == 403:
        return (
            f"{provider} denied access (HTTP 403). Check API access and key restrictions."
        )
    if status_code == 404:
        return (
            f"{provider} could not find the configured model or endpoint (HTTP 404). "
            "Check BANKAI_LLM_MODEL and BANKAI_LLM_BASE_URL."
        )
    if status_code == 429:
        return (
            f"{provider} rate limit or quota was reached (HTTP 429). "
            "Retry later; deterministic mock routing is available."
        )
    if status_code >= 500:
        return (
            f"{provider} is temporarily unavailable (HTTP {status_code}). "
            "Retry later; deterministic mock routing is available."
        )
    return f"The LLM provider returned HTTP {status_code}."


def _normalize_json_text(raw_text: str) -> str:
    """Normalize raw model response string to extract a clean JSON payload candidate."""
    text = raw_text.strip()
    if not text:
        return text

    import re
    text = _strip_safety_preambles(text)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1].strip()

    return text


def classify_question(message: str, settings: LLMSettings) -> RoutingPlan:
    """Ask the configured provider for a structured route and validate it locally."""
    content = ""
    cleaned = ""
    first_exception: Exception | None = None
    try:
        content = _request_completion(message, settings)
        cleaned = _normalize_json_text(content)
        output = json.loads(cleaned)
        return validate_route_output(output)
    except (json.JSONDecodeError, LLMError) as first_err:
        first_exception = first_err
        err_msg = str(first_err)
        if isinstance(first_err, LLMError) and any(
            pattern in err_msg
            for pattern in (
                "HTTP",
                "timed out",
                "could not be reached",
                "rejected API-key",
                "denied access",
                "rate limit",
                "not configured",
            )
        ):
            raise first_err

    retry_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": message},
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                "CRITICAL: Your previous output was invalid. "
                "Return EXACTLY ONE raw JSON object matching this schema: "
                '{"agents": [], "banking_actions": []}. '
                'Loan-only example: {"agents": ["loan"], "banking_actions": []}. '
                'Fraud-only example: {"agents": ["fraud"], "banking_actions": []}. '
                'Use banking_actions only for balance or transactions when "banking" is selected. '
                "No markdown code fences, no preamble text, no explanations."
            ),
        },
    ]
    try:
        retry_content = _request_completion(
            message, settings, messages_override=retry_messages
        )
        retry_cleaned = _normalize_json_text(retry_content)
        retry_output = json.loads(retry_cleaned)
        return validate_route_output(retry_output)
    except Exception:
        if isinstance(first_exception, LLMError):
            raise first_exception
        snippet = (cleaned[:60] + "...") if len(cleaned) > 60 else cleaned
        raise LLMError(f"The LLM response was not valid JSON (received: {snippet!r}).") from None
