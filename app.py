"""Streamlit interface for BankAI Multi-Agent Banking Assistant."""

from __future__ import annotations

import streamlit as st

from bankai import storage
from bankai.agents import AssistantResult, supervisor_agent


st.set_page_config(
    page_title="BankAI | Multi-Agent Banking Assistant",
    page_icon="B",
    layout="wide",
)

storage.initialize_database()
customers = storage.list_customers()
customer_options = {
    f"{customer['name']} · {customer['customer_id']}": customer["customer_id"]
    for customer in customers
}

st.title("BankAI")
st.caption("Multi-Agent Banking Assistant · Fictional data · Deterministic demo mode")

with st.sidebar:
    st.subheader("Demo workspace")
    selected_label = st.selectbox("Customer profile", list(customer_options))
    selected_customer_id = customer_options[selected_label]
    st.info("Offline mock mode is active. No API key or real bank connection is used.")
    st.divider()
    st.markdown("**Available agents**")
    st.markdown("- Supervisor Agent")
    st.markdown("- Banking Agent")
    st.markdown("- Fraud Agent")
    st.markdown("- Loan Agent")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if st.session_state.get("_active_customer_id") != selected_customer_id:
    st.session_state["_active_customer_id"] = selected_customer_id
    st.session_state["messages"] = []

selected_customer = next(
    customer for customer in customers if customer["customer_id"] == selected_customer_id
)
col_balance, col_account, col_profile = st.columns(3)
col_balance.metric("Available balance", f"${selected_customer['balance']:,.2f}")
col_account.metric("Demo account", selected_customer["account_id"])
col_profile.metric("Selected profile", selected_customer["name"])

st.subheader("Ask your banking assistant")
st.caption("The supervisor selects one or more agents, then shows exactly what ran.")

demo_use_cases = [
    ("Balance", "What's my current account balance?"),
    ("Transactions", "Show my 5 most recent transactions."),
    ("Fraud risk", "Is a $950 online purchase in Singapore suspicious?"),
    ("Loan + EMI", "Would I qualify for a $25,000 loan over 36 months? Calculate the EMI."),
    (
        "Multi-agent",
        "Check my balance, assess risk for a $2,400 online purchase, and estimate EMI "
        "for a $30,000 loan over 48 months.",
    ),
]

st.markdown("**Try a use case**")
use_case_columns = st.columns(len(demo_use_cases))


def run_prompt(prompt: str) -> None:
    result: AssistantResult = supervisor_agent(selected_customer_id, prompt)
    st.session_state.setdefault("messages", []).extend(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": result.answer,
                "trace": result.trace,
            },
        ]
    )


for column, (label, prompt) in zip(use_case_columns, demo_use_cases):
    if column.button(label, use_container_width=True, key=f"demo_{label}"):
        run_prompt(prompt)
        st.rerun()

st.divider()
for index, message in enumerate(st.session_state.get("messages", [])):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("trace"):
            with st.expander("Agent execution trace", expanded=False):
                for step_number, step in enumerate(message["trace"], start=1):
                    st.markdown(
                        f"**{step_number}. {step['agent']}** · {step['status']}  \n"
                        f"{step['detail']}"
                    )

prompt = st.chat_input(
    "Ask about a balance, transaction, fraud risk, or loan eligibility…"
)
if prompt:
    run_prompt(prompt)
    st.rerun()
