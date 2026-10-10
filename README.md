# BankAI Multi-Agent Banking Assistant

A small, runnable Streamlit internship demo of a supervisor coordinating banking,
fraud, and loan agents. All customers, account data, transactions, risk scores, and
loan decisions are fictional and simulated.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.port 5000
```

The first launch creates `bankai.db` in the project folder and seeds three demo
customers. No API key, cloud database, or bank connection is required; without an
LLM key the app uses its deterministic mock router.

## Optional live LLM intent routing

The LLM is used only to interpret the question and return a structured list of
agents and Banking Agent actions. The application validates that response before
routing. It does not send customer IDs, balances, or transaction records to the
provider. SQLite remains the source for account data, while fraud risk and loan
eligibility/EMI are still calculated by deterministic Python code.

Configure the key as a secret environment variable named `BANKAI_LLM_API_KEY`.
In Replit, add it through the Secrets pane; set the non-secret provider options
as environment variables. Locally, inject the key through your environment or
secret manager. Do not put the key in source code or commit it. Restart the Run
workflow after changing or replacing the key so the app process reads the new
value.
Without this variable, the application stays in working `MOCK MODE`. Invalid
model output, request timeouts, and provider errors also fall back to the mock
router and are labeled as `MOCK MODE` in the response.

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `BANKAI_LLM_API_KEY` | Secret provider key; enables live routing | Not set |
| `BANKAI_LLM_PROVIDER` | `openai`, `openrouter`, or `custom` | `openai` |
| `BANKAI_LLM_MODEL` | Provider model name | `gpt-4o-mini` for OpenAI |
| `BANKAI_LLM_BASE_URL` | OpenAI-compatible API root; required for `custom` | Provider default |
| `BANKAI_LLM_TIMEOUT_SECONDS` | Request timeout, from 1 to 60 seconds | `10` |

OpenAI and OpenRouter use their standard API roots by default. For another
OpenAI-compatible provider, set `BANKAI_LLM_PROVIDER=custom` and configure
`BANKAI_LLM_BASE_URL` and `BANKAI_LLM_MODEL`. HTTPS is required, except for
localhost endpoints. Each assistant response reports whether it used live LLM
routing or mock routing.

### Gemini on Replit

Keep the API key in `BANKAI_LLM_API_KEY` and configure these non-secret values:

```text
BANKAI_LLM_PROVIDER=custom
BANKAI_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
BANKAI_LLM_MODEL=gemini-3.8-flash
BANKAI_LLM_TIMEOUT_SECONDS=10
```

The app sends a Bearer-authenticated OpenAI-compatible `chat/completions`
request with JSON response formatting. For Gemini, it sets low reasoning and a
1024-token output cap so the model can return the validated route. Gemini's
current response for this project reports that
`gemini-2.5-flash` is unavailable to new users and recommends `gemini-3.8-flash`.
Use a Gemini model ID enabled for the API key. If authentication fails, the UI
shows a safe 401 message without exposing the key and continues with mock mode;
429 responses identify rate/quota limits and also fall back to mock routing.

## Try the five use cases

1. Ask: “What's my current account balance?”
2. Ask: “Show my 5 most recent transactions.”
3. Ask: “Is a $950 online purchase in Singapore suspicious?”
4. Ask: “Would I qualify for a $25,000 loan over 36 months? Calculate the EMI.”
5. Ask: “Check my balance, assess risk for a $2,400 online purchase, and estimate
   EMI for a $30,000 loan over 48 months.” This routes to multiple specialist agents.

Use the customer selector to try different simulated customer profiles. The
execution trace records the supervisor and each specialist that actually ran.

## Run tests

```bash
python -m unittest discover -v
```

These automated tests are offline: they use fake provider responses and cover
intent routing, multi-agent execution, SQLite-backed balances/transactions,
deterministic fraud and loan calculations, and fallback for invalid output,
timeouts, authentication errors, unavailable models, and rate limits. They do
not verify access to a live Gemini account; live verification requires using the
configured Replit workflow and a real request.

For this workspace, the latest verification passed 16 automated tests and 6
separate live Gemini checks: one minimal route plus balance, recent transactions,
fraud risk, loan eligibility/EMI, and a multi-agent request. The live checks are
not part of `unittest` and should be repeated after changing the key or model.

## How the demo works

- **Supervisor Agent** uses validated structured LLM intent routing when configured;
  otherwise it uses the deterministic mock router. It can coordinate multiple
  specialist agents in one request.
- **Banking Agent** reads account balances and transaction history from SQLite.
- **Fraud Agent** applies a stable rule-based formula to the requested amount and
  stated transaction signals; its score is not a bank decision.
- **Loan Agent** applies simplified simulated eligibility rules and estimates a
  monthly EMI using a fixed 11.5% annual rate.
- `bankai/storage.py` owns the schema, fictional seed records, and read helpers.
- `bankai/agents.py` contains routing, banking, fraud, and loan logic.
- `app.py` provides the chat dashboard and five ready-to-run example requests.

This project is an educational prototype. It does not provide financial advice,
make real lending decisions, connect to a bank, or detect actual fraud.
