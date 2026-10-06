# 🗄️ Project 03 — SQL Database Agent

Ask questions in plain English; a **LangGraph** workflow writes real SQL,
**refuses destructive statements**, executes the query against SQLite, and
explains the result. The generated SQL, the result table and the explanation are
all shown for every question.

## Pipeline

```
Question → Schema introspection → Generate SQL → Guard (safe?) ─┬─ yes → Execute → Explain → Answer
                                                                 └─ no  → Refuse
```

## Features

- **Real text-to-SQL**: the LLM writes SQLite SQL from the live schema (LangChain
  `SQLDatabase` introspection + few-shot prompting)
- **Safety guardrails**: only a single read-only `SELECT` / `WITH` query is
  allowed; `INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA/…` are rejected before
  execution, and string literals are ignored when scanning keywords
- **Row cap**: queries are auto-limited (≤ 200 rows) to keep results sane
- **Explanation**: results are explained in plain language
- **Synthetic demo data**: deterministic 4-table e-commerce DB
  (`customers`, `products`, `orders`, `order_items`)

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (entry point) |
| `database.py` | Builds the demo SQLite DB + LangChain `SQLDatabase` wrapper |
| `agent.py` | SQL generation, guardrails, execution, explanation |
| `graph.py` | LangGraph workflow with conditional safe/unsafe routing |
| `llm.py` | Groq LLM wiring + robust text extraction |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Streamlit runtime config |

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

| Setting | Value |
|---------|-------|
| Repository | `deepvisionkararhaider-crypto/langchain-langgraph-ai-lab` |
| Branch | `main` |
| Main file path | `03_sql_agent/app.py` |

**Required secrets:**

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## Try these questions

- How many customers are on each plan?
- What are the top 5 product categories by total revenue?
- Which country has the highest average order value?
- How many orders were refunded in 2024?

## Notes / limitations

- The demo database is rebuilt fresh on each session (in-memory SQLite).
- Only read-only queries are permitted by design — this is an agent demo, not a
  data-editing tool.

**Status:** ✅ Built · ✅ Tested locally · 🔜 Ready for Streamlit deployment
