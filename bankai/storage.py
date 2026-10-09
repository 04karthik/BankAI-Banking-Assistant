"""SQLite storage and fictional banking seed data for BankAI."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT_DIR / "bankai.db"

CUSTOMERS = (
    {
        "customer_id": "CUST-1001",
        "name": "Alex Morgan",
        "monthly_income": 6200.0,
        "credit_score": 742,
        "monthly_debt": 550.0,
        "account_id": "BA-4821",
        "balance": 18420.65,
    },
    {
        "customer_id": "CUST-1002",
        "name": "Jamie Chen",
        "monthly_income": 4100.0,
        "credit_score": 681,
        "monthly_debt": 460.0,
        "account_id": "BA-7390",
        "balance": 9325.18,
    },
    {
        "customer_id": "CUST-1003",
        "name": "Taylor Rivera",
        "monthly_income": 2950.0,
        "credit_score": 598,
        "monthly_debt": 780.0,
        "account_id": "BA-1156",
        "balance": 2760.42,
    },
)

TRANSACTION_SEEDS = (
    ("Northstar Market", "Groceries", -86.42, "Seattle, US", "Completed"),
    ("Harbor Coffee", "Dining", -8.75, "Seattle, US", "Completed"),
    ("Monthly salary", "Income", 6200.00, "Direct deposit", "Completed"),
    ("Metro Transit", "Transport", -32.50, "Seattle, US", "Completed"),
    ("Streamline Media", "Subscription", -14.99, "Online", "Completed"),
    ("Evergreen Pharmacy", "Health", -28.13, "Seattle, US", "Completed"),
    ("Cedar & Stone", "Dining", -54.60, "Portland, US", "Completed"),
    ("Cloud Storage", "Subscription", -2.99, "Online", "Completed"),
)


def connect(db_path: str | Path = DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection that returns rows by column name."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path = DATABASE_PATH) -> None:
    """Create the schema and seed fictional data once."""
    with connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                monthly_income REAL NOT NULL,
                credit_score INTEGER NOT NULL,
                monthly_debt REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                account_type TEXT NOT NULL,
                currency TEXT NOT NULL,
                balance REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                posted_at TEXT NOT NULL,
                merchant TEXT NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                location TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )

        existing = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if existing:
            return

        today = date.today()
        for customer_index, customer in enumerate(CUSTOMERS):
            connection.execute(
                """
                INSERT INTO customers
                    (customer_id, name, monthly_income, credit_score, monthly_debt)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    customer["customer_id"],
                    customer["name"],
                    customer["monthly_income"],
                    customer["credit_score"],
                    customer["monthly_debt"],
                ),
            )
            connection.execute(
                """
                INSERT INTO accounts
                    (account_id, customer_id, account_type, currency, balance)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    customer["account_id"],
                    customer["customer_id"],
                    "Everyday Checking",
                    "USD",
                    customer["balance"],
                ),
            )

            for transaction_index, seed in enumerate(TRANSACTION_SEEDS):
                merchant, category, amount, location, status = seed
                posted_at = today - timedelta(days=transaction_index + customer_index)
                connection.execute(
                    """
                    INSERT INTO transactions
                        (transaction_id, customer_id, posted_at, merchant, category,
                         amount, location, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"TX-{customer_index + 1:02d}-{transaction_index + 1:03d}",
                        customer["customer_id"],
                        posted_at.isoformat(),
                        merchant,
                        category,
                        amount if customer_index == 0 or category != "Income" else amount * 0.85,
                        location,
                        status,
                    ),
                )


def get_customer(
    customer_id: str, db_path: str | Path = DATABASE_PATH
) -> dict[str, Any] | None:
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    return dict(row) if row else None


def get_account(
    customer_id: str, db_path: str | Path = DATABASE_PATH
) -> dict[str, Any] | None:
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM accounts WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    return dict(row) if row else None


def get_transactions(
    customer_id: str, limit: int = 5, db_path: str | Path = DATABASE_PATH
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 50))
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT transaction_id, posted_at, merchant, category, amount, location, status
            FROM transactions
            WHERE customer_id = ?
            ORDER BY posted_at DESC, transaction_id DESC
            LIMIT ?
            """,
            (customer_id, safe_limit),
        ).fetchall()
    return [dict(row) for row in rows]


def list_customers(db_path: str | Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT c.customer_id, c.name, a.account_id, a.balance, a.currency
            FROM customers AS c
            JOIN accounts AS a ON a.customer_id = c.customer_id
            ORDER BY c.name
            """
        ).fetchall()
    return [dict(row) for row in rows]
