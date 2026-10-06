"""Project 03 — SQL agent as a LangGraph.

    START → introspect → generate_sql → guard ─┬─(safe)──► execute → explain → END
                                                └─(unsafe)► refuse  ─────────► END

Conditional routing means an unsafe query is stopped *before* execution and a
friendly refusal is returned instead.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import (
    SQLResult,
    enforce_limit,
    explain_result,
    generate_sql,
    is_safe,
    run_sql,
)


class SQLState(TypedDict, total=False):
    question: str
    max_rows: int
    sql: str
    safe: bool
    reason: str
    columns: list[str]
    rows: list[tuple]
    explanation: str
    error: str
    steps: Annotated[list[str], lambda a, b: (a or []) + (b or [])]


def build_graph(connection: sqlite3.Connection, db):
    def introspect(state: SQLState) -> dict:
        try:
            db.get_table_info()
            return {"steps": ["introspect: schema loaded"]}
        except Exception as exc:
            return {"error": f"introspection failed: {exc}", "steps": ["introspect: error"]}

    def gen(state: SQLState) -> dict:
        try:
            sql = generate_sql(db, state["question"])
            return {"sql": sql, "steps": ["generate_sql: query drafted"]}
        except Exception as exc:
            return {"sql": "", "error": f"generation failed: {exc}", "steps": ["generate_sql: error"]}

    def guard(state: SQLState) -> dict:
        safe, reason = is_safe(state.get("sql", ""))
        return {"safe": safe, "reason": reason, "steps": [f"guard: {'safe' if safe else reason}"]}

    def route(state: SQLState) -> str:
        return "execute" if state.get("safe") else "refuse"

    def execute(state: SQLState) -> dict:
        try:
            sql = enforce_limit(state["sql"], int(state.get("max_rows", 200)))
            cols, rows = run_sql(connection, sql, int(state.get("max_rows", 200)))
            return {"sql": sql, "columns": cols, "rows": rows, "steps": [f"execute: {len(rows)} row(s)"]}
        except Exception as exc:
            return {"error": f"execution failed: {exc}", "steps": ["execute: error"]}

    def explain(state: SQLState) -> dict:
        try:
            text = explain_result(
                state["question"], state.get("sql", ""), state.get("columns", []), state.get("rows", [])
            )
        except Exception as exc:
            text = f"(explanation unavailable: {exc})"
        return {"explanation": text, "steps": ["explain: done"]}

    def refuse(state: SQLState) -> dict:
        return {
            "explanation": (
                "I refused to run that query because it is not a read-only SELECT "
                f"({state.get('reason', 'unsafe')})."
            ),
            "steps": ["refuse: unsafe query blocked"],
        }

    g = StateGraph(SQLState)
    g.add_node("introspect", introspect)
    g.add_node("generate_sql", gen)
    g.add_node("guard", guard)
    g.add_node("execute", execute)
    g.add_node("explain", explain)
    g.add_node("refuse", refuse)
    g.add_edge(START, "introspect")
    g.add_edge("introspect", "generate_sql")
    g.add_edge("generate_sql", "guard")
    g.add_conditional_edges("guard", route, {"execute": "execute", "refuse": "refuse"})
    g.add_edge("execute", "explain")
    g.add_edge("explain", END)
    g.add_edge("refuse", END)
    return g.compile()


def ask(connection: sqlite3.Connection, db, question: str, max_rows: int = 200) -> SQLResult:
    """Run the LangGraph SQL workflow and return a populated SQLResult."""
    graph = build_graph(connection, db)
    state: SQLState = {"question": question, "max_rows": max_rows}
    out: dict[str, Any] = graph.invoke(state)
    return SQLResult(
        question=question,
        sql=out.get("sql", ""),
        safe=out.get("safe", False),
        reason=out.get("reason", "ok"),
        columns=out.get("columns", []),
        rows=out.get("rows", []),
        explanation=out.get("explanation", ""),
        error=out.get("error", ""),
    )
