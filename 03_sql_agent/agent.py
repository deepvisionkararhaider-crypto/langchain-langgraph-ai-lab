"""Project 03 — SQL Database Agent: NL → SQL → execute → explain.

A real text-to-SQL agent:
  1. introspect the schema with LangChain ``SQLDatabase``
  2. ask the LLM to write a single read-only SQLite query (few-shot guided)
  3. **guard** the query (reject anything that is not a lone SELECT/WITH)
  4. execute it against SQLite
  5. explain the result in plain language

Destructive statements (INSERT/UPDATE/DELETE/DROP/ALTER/…) are refused before
they ever reach the database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import text as _sql_text

from llm import extract_text, get_llm

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|grant|revoke|into)\b",
    re.IGNORECASE,
)
ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def clean_sql(text: str) -> str:
    """Strip markdown fences / prose and return a single SQL statement."""
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    # Drop a trailing semicolon and anything after the first statement.
    text = text.split(";")[0].strip()
    return text


def is_safe(sql: str) -> tuple[bool, str]:
    """Validate that ``sql`` is a single read-only SELECT/WITH query."""
    if not sql:
        return False, "empty query"
    if not ALLOWED_START.match(sql):
        return False, "only SELECT / WITH queries are allowed"
    # ignore string literals when scanning for forbidden keywords
    without_strings = re.sub(r"'[^']*'", "''", sql)
    if FORBIDDEN.search(without_strings):
        return False, "destructive or schema-modifying keywords detected"
    if without_strings.count("(") != without_strings.count(")"):
        return False, "unbalanced parentheses"
    return True, "ok"


def enforce_limit(sql: str, max_rows: int = 200) -> str:
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {max_rows}"


def run_sql(engine, sql: str, max_rows: int = 200) -> tuple[list[str], list[tuple]]:
    """Execute a read-only query and return (columns, rows)."""
    with engine.connect() as conn:
        result = conn.execute(_sql_text(sql))
        cols = list(result.keys())
        rows = [tuple(r) for r in result.fetchmany(max_rows + 1)]
    return cols, rows[:max_rows]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
SQL_PROMPT = """You are an expert SQLite analyst. Write ONE read-only SQL query that
answers the user's question using only the schema below.

Rules:
- Output ONLY the SQL. No explanation, no markdown fences.
- Use only SELECT (or a WITH ... SELECT ...) statements. Never modify data.
- Use the exact table and column names from the schema.
- Prefer explicit column aliases and ORDER BY for clarity.
- Dates are stored as TEXT 'YYYY-MM-DD'. Use functions like strftime when needed.
- If a LIMIT is helpful, include one (<= 200 rows).

Schema:
{schema}

{examples}
Question: {question}

SQL:"""

EXPLAIN_PROMPT = """You are a data analyst. Explain to a non-technical user what the query
did and what the results mean. Be concise (3-6 sentences or short bullets). Do not
invent numbers that are not present in the result. If the result is empty, say so.

Question: {question}

SQL:
{sql}

Columns: {columns}
Row count: {n}
Rows (up to 15): {rows}

Explanation:"""

FEWSHOT = """Examples:
Q: How many customers are on the 'pro' plan?
SQL: SELECT COUNT(*) AS pro_customers FROM customers WHERE plan = 'pro';
Q: What are the top 5 product categories by total revenue?
SQL: SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.category ORDER BY revenue DESC LIMIT 5;
"""


@dataclass
class SQLResult:
    question: str
    sql: str = ""
    safe: bool = True
    reason: str = "ok"
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    explanation: str = ""
    error: str = ""


def generate_sql(db, question: str) -> str:
    llm = get_llm(temperature=0.0, max_tokens=512)
    prompt = SQL_PROMPT.format(
        schema=db.get_table_info(), examples=FEWSHOT, question=question
    )
    return clean_sql(extract_text(llm.invoke(prompt)))


def explain_result(question: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
    llm = get_llm(temperature=0.2, max_tokens=600)
    preview = rows[:15]
    prompt = EXPLAIN_PROMPT.format(
        question=question, sql=sql, columns=columns, n=len(rows), rows=preview
    )
    return extract_text(llm.invoke(prompt)).strip()


def answer(engine, db, question: str, max_rows: int = 200) -> SQLResult:
    """Full pipeline: generate → guard → execute → explain."""
    result = SQLResult(question=question)
    try:
        sql = generate_sql(db, question)
    except Exception as exc:
        result.error = f"SQL generation failed: {exc}"
        return result

    safe, reason = is_safe(sql)
    result.sql = sql
    result.safe = safe
    result.reason = reason
    if not safe:
        result.error = f"Refused unsafe query ({reason})."
        return result

    try:
        cols, rows = run_sql(engine, enforce_limit(sql, max_rows), max_rows)
        result.columns, result.rows = cols, rows
    except Exception as exc:
        result.error = f"Execution failed: {exc}"
        return result

    try:
        result.explanation = explain_result(question, sql, cols, rows)
    except Exception as exc:
        result.explanation = f"(explanation unavailable: {exc})"
    return result
