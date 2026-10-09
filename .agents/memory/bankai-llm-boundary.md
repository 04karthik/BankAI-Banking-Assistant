---
name: BankAI LLM boundary
description: The required separation between optional model intent routing and simulated banking data/calculations.
---

For BankAI, an optional LLM may interpret a question and return validated agent/action labels only. Account balances and transaction history must come from SQLite; fraud risk, eligibility, and EMI must remain in deterministic Python logic.

**Why:** The user requested optional live intent routing while keeping fictional banking records and specialist calculations local and reproducible.

**How to apply:** Validate all model-produced routing data before dispatch. Never treat model output as a source of banking records or a financial decision.
