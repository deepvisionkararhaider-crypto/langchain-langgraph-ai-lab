"""Project 02 — Multi-Agent Researcher (Streamlit entry point).

A LangGraph team of four agents (Research → Fact → Analysis → Writer) answers a
question using **real** web search + page extraction. Sources are the actual
URLs that were fetched — never fabricated.

Run:  streamlit run 02_multi_agent_researcher/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import run_research  # noqa: E402
from llm import get_secret, missing_key_message  # noqa: E402

st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🔎", layout="wide")

AGENT_ICONS = {
    "ResearchAgent": "🔍",
    "FactAgent": "🧾",
    "AnalysisAgent": "🧠",
    "WriterAgent": "✍️",
}


def _sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        depth = st.slider("Sub-queries (research breadth)", 1, 5, 3)
        max_sources = st.slider("Sources per sub-query", 2, 6, 4)
        st.divider()
        st.caption(f"LLM: {get_secret('GROQ_MODEL', 'openai/gpt-oss-120b')} (Groq)")
        st.caption("Search: DuckDuckGo (live)")
        st.divider()
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.pop("last", None)
            st.rerun()
    return depth, max_sources


def _render_pipeline(trace, facts, analysis, sources, answer):
    st.subheader("🤖 Agent pipeline")
    for step in trace:
        icon = AGENT_ICONS.get(step.get("agent"), "•")
        st.markdown(f"{icon} **{step.get('agent')}** — {step.get('message')}")

    if facts:
        with st.expander(f"🧾 Extracted facts ({len(facts)})", expanded=False):
            for f in facts:
                tag = f" [S{f['source']}]" if f.get("source") else ""
                st.markdown(f"- {f['claim']}{tag}")

    if analysis:
        with st.expander("🧠 Analysis brief", expanded=False):
            st.markdown(analysis)

    st.subheader("📝 Final answer")
    st.markdown(answer or "_(no answer produced)_")

    if sources:
        st.subheader(f"🔗 Sources ({len(sources)})")
        for s in sources:
            st.markdown(f"**[{s['tag']}]** {s['title']} — [{s['url']}]({s['url']})")


def main():
    st.title("🔎 Multi-Agent Researcher")
    st.caption("LangGraph · four specialised agents · real web search + page extraction")

    if not get_secret("GROQ_API_KEY"):
        st.warning(missing_key_message())

    depth, max_sources = _sidebar()

    question = st.text_area(
        "Research question",
        placeholder="e.g. How does LangGraph differ from LangChain, and when should each be used?",
        height=90,
    )
    run = st.button("🚀 Run research", type="primary", disabled=not question.strip())

    if run:
        with st.spinner("Researching the web and coordinating agents…"):
            result = run_research(question, depth=depth, max_sources=max_sources)
        st.session_state["last"] = result

    last = st.session_state.get("last")
    if last:
        if last.get("subqueries"):
            st.info("**Planned queries:** " + " · ".join(f"`{q}`" for q in last["subqueries"]))
        if last.get("error"):
            st.warning(f"Note: {last['error']}")
        if not last.get("sources"):
            st.warning(
                "No web sources were reachable from this environment. The pipeline ran, "
                "but the answer is ungrounded — try again or check outbound network access."
            )
        _render_pipeline(
            last.get("trace", []),
            last.get("facts", []),
            last.get("analysis", ""),
            last.get("sources", []),
            last.get("answer", ""),
        )

        with st.expander("📥 Export"):
            st.download_button(
                "Download markdown report",
                data=_to_markdown(question, last),
                file_name="research_report.md",
                mime="text/markdown",
            )
    else:
        st.info("Enter a question and press **Run research** to start the agent pipeline.")


def _to_markdown(question: str, r: dict) -> str:
    lines = [f"# Research report\n\n**Question:** {question}\n", "## Answer\n", r.get("answer", ""), "\n## Sources\n"]
    for s in r.get("sources", []):
        lines.append(f"- [{s['tag']}] {s['title']} — {s['url']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
