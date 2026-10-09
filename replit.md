# BankAI Multi-Agent Banking Assistant

A Streamlit internship demo where a supervisor routes fictional banking questions to specialized agents backed by SQLite.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `streamlit run app.py --server.port 5000` — run the BankAI dashboard
- `python -m unittest discover -v` — test the BankAI agents and SQLite tools

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Python 3.10+, Streamlit, and SQLite for the BankAI demo
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

_Populate as you build — short repo map plus pointers to the source-of-truth file for DB schema, API contracts, theme files, etc._

## Architecture decisions

_Populate as you build — non-obvious choices a reader couldn't infer from the code (3-5 bullets)._

## Product

BankAI demonstrates supervisor routing, account balance and transaction lookups, deterministic transaction-risk scoring, and simulated loan eligibility with EMI estimates. It uses fictional data and works offline without an API key.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

_Populate as you build — sharp edges, "always run X before Y" rules._

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
