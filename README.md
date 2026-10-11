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
customers. With no cloud API key configured, BankAI uses the local Ollama model
for intent routing. If Ollama is unavailable or returns an error, the app uses
its deterministic mock router. No cloud database or bank connection is required.

### Windows setup (PowerShell)

Run these commands from the project folder. Python 3.10 or newer and the Python
launcher (`py`) should be installed.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py --server.port 5000
```

### Local Ollama on Windows

If Ollama is not installed, install it first. Then open PowerShell in the
project folder and download the local model:

```powershell
winget install --id Ollama.Ollama -e
ollama pull llama3.2:latest
ollama list
```

Start the Ollama application if it is not already running, then launch BankAI.
No provider variables or API key are needed for the local default:

```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.port 5000
```

Ollama uses `http://localhost:11434/v1` and model `llama3.2` by default. It does
not need an API key; BankAI sends requests only to a loopback address for this
provider and does not send an Authorization header. Ollama uses a 60-second
default timeout for local model startup. If cloud API keys are present and you
want to force local routing, set `$env:BANKAI_LLM_PROVIDER = "ollama"` in that
PowerShell window before launching the app.

## Optional live LLM intent routing

The LLM is used only to interpret the question and return a structured list of
agents and Banking Agent actions. The application validates that response before
routing. It does not send customer IDs, balances, or transaction records to the
provider. SQLite remains the source for account data, while fraud risk and loan
eligibility/EMI are still calculated by deterministic Python code.

For cloud providers, configure the key as a secret environment variable named
`BANKAI_LLM_API_KEY`. In Replit, add it through the Secrets pane; set the
non-secret provider options as environment variables. Locally, inject the key
through your environment or secret manager. Do not put the key in source code or
commit it. Without cloud API keys, the application uses local Ollama by default.
Invalid model output, request timeouts, and provider errors fall back to the
deterministic router and are labeled as `MOCK MODE` in the response. Restart the
Run workflow after changing or replacing a cloud key so the app process reads
the new value.

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `BANKAI_LLM_API_KEY` | Secret provider key; required for remote providers | Not set; not required for local Ollama |
| `BANKAI_LLM_PROVIDER` | `openai`, `openrouter`, `groq`, `custom`, or `ollama` | `ollama` when no cloud key is configured |
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
