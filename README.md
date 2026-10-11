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

### Windows setup (PowerShell)

Run these commands from the project folder. Python 3.10 or newer and the Python
launcher (`py`) should be installed.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py --server.port 5000
```

### Optional local Ollama on Windows

Install Ollama for Windows, then open PowerShell in the project folder and
download a local model:

```powershell
winget install --id Ollama.Ollama -e
ollama pull llama3.2
```

Start the Ollama application if it is not already running. Then select its local
OpenAI-compatible endpoint and launch BankAI:

```powershell
$env:BANKAI_LLM_PROVIDER = "ollama"
$env:BANKAI_LLM_BASE_URL = "http://localhost:11434/v1"
$env:BANKAI_LLM_MODEL = "llama3.2"
.\.venv\Scripts\streamlit.exe run app.py --server.port 5000
```

Ollama defaults to `http://localhost:11434/v1` and does not need an API key. The
application accepts the keyless Ollama provider only when its endpoint is a
loopback address. Ollama uses a 60-second default timeout for local model
startup. If Ollama is unavailable, returns invalid output, or its route conflicts
with clear supported keywords, BankAI continues using deterministic local
routing. The environment variables above apply to the current PowerShell window
only.

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
| `BANKAI_LLM_API_KEY` | Secret provider key; required for remote providers | Not set; not required for local Ollama |
| `BANKAI_LLM_PROVIDER` | `openai`, `openrouter`, `groq`, `custom`, or `ollama` | `openai` |
| `BANKAI_LLM_MODEL` | Provider model name | Provider default (`llama3.2` for Ollama) |
| `BANKAI_LLM_BASE_URL` | OpenAI-compatible API root; required for `custom` | Provider default (`http://localhost:11434/v1` for Ollama) |
| `BANKAI_LLM_TIMEOUT_SECONDS` | Request timeout, from 1 to 60 seconds | `10` (`60` for Ollama) |

OpenAI and OpenRouter use their standard API roots by default. For another
OpenAI-compatible provider, set `BANKAI_LLM_PROVIDER=custom` and configure
`BANKAI_LLM_BASE_URL` and `BANKAI_LLM_MODEL`. HTTPS is required, except for
localhost endpoints. Keyless mode is limited to the local Ollama provider;
OpenAI, OpenRouter, Groq, and custom remote providers still require an API key.
Each assistant response reports whether it used live LLM routing or mock
routing.

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
5. Ask: “Check my balance; assess fraud risk for a $2,400 online purchase;
  calculate eligibility and EMI for a $30,000 loan over 48 months.” This routes
  to Banking, Fraud, and Loan agents in sequence and returns separate sections.

Use the customer selector to try different simulated customer profiles. The
execution trace records the supervisor and each specialist that actually ran.

## Run tests

```bash
python -m unittest discover -v
```

These automated tests are offline: they use fake provider responses and cover
intent routing, multi-agent execution, SQLite-backed balances/transactions,
deterministic fraud and loan calculations, UI button/session behavior, and
fallback for invalid output, timeouts, authentication errors, unavailable
models, and rate limits. Run them with the active environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

The automated suite mocks LLM responses. For a live local Ollama check, start
Ollama, configure the provider as shown above, and exercise all five examples in
the Streamlit app.

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
- `app.py` provides the Streamlit chat dashboard and five ready-to-run example
  requests. In the Replit workspace, the web artifact routes its service to this
  root Streamlit app; the tracked React files are part of that workspace setup.

This project is an educational prototype. It does not provide financial advice,
make real lending decisions, connect to a bank, or detect actual fraud.
