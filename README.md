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
customers. No API key, cloud database, or bank connection is required.

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

## How the demo works

- **Supervisor Agent** applies deterministic intent routing and coordinates one or
  more specialist agents.
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
