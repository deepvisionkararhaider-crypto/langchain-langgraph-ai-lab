"""langchain-langgraph-ai-lab — portfolio dashboard (Streamlit root entry point).

A lightweight launcher that describes every project and shows exactly how to run
or deploy it. It does **not** import the individual projects' heavy modules, so
it starts instantly and stays cheap (important on Streamlit Community Cloud).

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="LangChain + LangGraph AI Lab", page_icon="🧪", layout="wide")

REPO = "deepvisionkararhaider-crypto/langchain-langgraph-ai-lab"

PROJECTS = [
    {
        "n": 1, "name": "RAG Document Chat", "icon": "📄", "path": "01_rag_document_chat",
        "desc": "Upload PDF/DOCX/TXT, chunk, embed, retrieve and ask questions with source-cited answers.",
        "tech": "LangChain · LangGraph · fastembed · FAISS · Groq",
        "graph": "retrieve → route → generate",
    },
    {
        "n": 2, "name": "Multi-Agent Researcher", "icon": "🔎", "path": "02_multi_agent_researcher",
        "desc": "Four collaborating agents plan sub-queries, gather real web evidence, extract facts and write a cited answer.",
        "tech": "LangGraph · LangChain · ddgs/Bing · Groq",
        "graph": "research → facts → analysis → writer",
    },
    {
        "n": 3, "name": "SQL Database Agent", "icon": "🗄️", "path": "03_sql_agent",
        "desc": "Natural language → SQL against SQLite, with destructive-statement guardrails and plain-language explanations.",
        "tech": "LangChain SQLDatabase · LangGraph · SQLAlchemy · Groq",
        "graph": "introspect → generate → guard → execute → explain",
    },
    {
        "n": 4, "name": "Web Research Agent", "icon": "🌐", "path": "04_web_research_agent",
        "desc": "Live web research pipeline that fetches pages, extracts verbatim passages and produces a cited summary.",
        "tech": "LangGraph · LangChain · requests/ddgs · Groq",
        "graph": "search → retrieve → extract → analyze → summarize",
    },
    {
        "n": 5, "name": "Customer Support Agent", "icon": "🎧", "path": "05_customer_support_agent",
        "desc": "Classifies intent, retrieves KB articles, drafts a grounded reply, QA-reviews it and escalates when unsure.",
        "tech": "LangGraph · LangChain · hybrid retrieval · Groq",
        "graph": "intent → retrieve → generate → review → finalize",
    },
    {
        "n": 6, "name": "Document Extraction Pipeline", "icon": "🧾", "path": "06_document_extraction_pipeline",
        "desc": "Parse documents and extract validated structured JSON (with downloadable output).",
        "tech": "LangChain · LangGraph · pydantic · Groq",
        "graph": "parse → extract → validate → structure → summarize",
    },
    {
        "n": 7, "name": "Planner + Executor Agent", "icon": "🧭", "path": "07_planner_executor_agent",
        "desc": "Plans a task list, executes steps, reviews results and loops via LangGraph conditional edges until done.",
        "tech": "LangGraph (conditional edges) · LangChain · Groq",
        "graph": "planner → executor → reviewer → (revise) → final",
    },
    {
        "n": 8, "name": "Code Review Agent", "icon": "🐍", "path": "08_code_review_agent",
        "desc": "Reviews pasted/uploaded Python for bugs, security and quality issues and suggests improved code.",
        "tech": "LangGraph · LangChain · AST static analysis · Groq",
        "graph": "static → bugs → security → quality → reviewer",
    },
    {
        "n": 9, "name": "Content Research Pipeline", "icon": "✍️", "path": "09_content_research_pipeline",
        "desc": "Turn a topic into researched, outlined, drafted, reviewed and improved content with tone/length control.",
        "tech": "LangGraph · LangChain · web research · Groq",
        "graph": "research → outline → draft → review → improve",
    },
    {
        "n": 10, "name": "Memory Personal Assistant", "icon": "🧠", "path": "10_memory_personal_assistant",
        "desc": "Chat assistant with persistent memory, preferences and context retrieval across sessions.",
        "tech": "LangGraph · LangChain · SQLite memory · Groq",
        "graph": "recall → agent → tools → respond → remember",
    },
]

DONE = {1, 2, 3, 4, 5}  # projects fully built, tested, pushed and verified


def _card(p: dict):
    status = "✅ Complete" if p["n"] in DONE else "🚧 Planned"
    with st.container(border=True):
        st.markdown(f"### {p['icon']} {p['n']:02d} · {p['name']}")
        st.markdown(p["desc"])
        st.caption(f"**Technologies:** {p['tech']}")
        st.caption(f"**Graph:** `{p['graph']}`")
        st.code(
            f"# local\nstreamlit run {p['path']}/app.py\n\n"
            f"# Streamlit Cloud main file\n{p['path']}/app.py",
            language="bash",
        )
        st.markdown(
            f"[📂 Source](https://github.com/{REPO}/tree/main/{p['path']})  ·  "
            f"**Status:** {status}"
        )


def main():
    st.title("🧪 LangChain + LangGraph AI Lab")
    st.markdown(
        "Ten real AI projects — each a self-contained **Streamlit** app powered by "
        "**LangChain** and **LangGraph**, deployed on **Streamlit Community Cloud**."
    )
    st.info(
        "**Deploy any project on Streamlit Community Cloud:** point the app at this "
        f"repo (`{REPO}`), branch `main`, and the project's `app.py` as the main file. "
        "Add `GROQ_API_KEY` in *Settings → Secrets*."
    )

    st.subheader(f"Projects — {len(DONE)}/10 complete & verified")
    st.progress(len(DONE) / 10)

    left, right = st.columns(2)
    for i, p in enumerate(PROJECTS):
        with (left if i % 2 == 0 else right):
            _card(p)

    st.divider()
    st.caption(
        "Each project has its own `app.py`, `requirements.txt` and `.streamlit/` config. "
        "API keys are read from Streamlit secrets or environment variables and are never committed."
    )


if __name__ == "__main__":
    main()
