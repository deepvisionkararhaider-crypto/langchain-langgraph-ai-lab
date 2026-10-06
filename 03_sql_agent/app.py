"""Project 03 — SQL Database Agent (Streamlit entry point).

Ask questions in plain English. A LangGraph workflow introspects the schema,
generates SQL, **guards against destructive statements**, executes it against
SQLite, and explains the result. Generated SQL, the result table and the
explanation are all shown.

Run:  streamlit run 03_sql_agent/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import build_connections, get_langchain_db, table_stats  # noqa: E402
from graph import ask  # noqa: E402
from llm import get_secret, missing_key_message  # noqa: E402

st.set_page_config(page_title="SQL Database Agent", page_icon="🗄️", layout="wide")


@st.cache_resource(show_spinner="Building demo database…")
def _resources():
    con = build_connections()
    db = get_langchain_db(con)
    return con, db


EXAMPLES = [
    "How many customers are on each plan?",
    "What are the top 5 product categories by total revenue?",
    "List the 10 most recent delivered orders with their totals.",
    "Which country has the highest average order value?",
    "How many orders were refunded in 2024?",
    "What is the average number of items per order?",
]


def main():
    st.title("🗄️ SQL Database Agent")
    st.caption("Natural language → SQL → guard → execute → explanation · LangChain + LangGraph")

    if not get_secret("GROQ_API_KEY"):
        st.warning(missing_key_message())

    con, db = _resources()

    with st.sidebar:
        st.header("📊 Demo database")
        for t, n in table_stats(con).items():
            st.metric(t, n)
        st.divider()
        st.header("🧭 Example questions")
        for q in EXAMPLES:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state["question"] = q
        st.divider()
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    with st.expander("🗂️ Database schema"):
        st.code(db.get_table_info(), language="sql")

    question = st.text_area(
        "Ask a question about the data",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Which products generated the most revenue in 2024?",
        height=80,
    )
    run = st.button("▶️ Run query", type="primary", disabled=not question.strip())

    if run:
        with st.spinner("Generating SQL, validating and executing…"):
            result = ask(con, db, question)
        history = st.session_state.setdefault("history", [])
        history.insert(0, result)

    for res in st.session_state.get("history", []):
        with st.container(border=True):
            st.markdown(f"#### ❓ {res.question}")

            if res.sql:
                st.markdown("**Generated SQL**")
                st.code(res.sql, language="sql")
                st.caption("✅ read-only query" if res.safe else f"⛔ blocked: {res.reason}")

            if res.error:
                st.error(res.error)

            if res.columns and res.rows:
                import pandas as pd

                df = pd.DataFrame(res.rows, columns=res.columns)
                st.dataframe(df, use_container_width=True, height=min(360, 40 + 28 * len(df)))
                st.caption(f"{len(res.rows)} row(s)")
            elif res.safe and not res.error:
                st.info("The query returned no rows.")

            if res.explanation:
                st.markdown("**Explanation**")
                st.markdown(res.explanation)

    if not st.session_state.get("history"):
        st.info("Pick an example from the sidebar or type your own question.")


if __name__ == "__main__":
    main()
