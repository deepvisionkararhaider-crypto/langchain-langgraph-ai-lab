"""Project 04 — Web Research Agent (Streamlit entry point).

A LangGraph pipeline that performs *real* web research:
search → retrieve → extract → analyze → summarize.

Every node is shown in the UI so you can follow the research trail, and the final
answer cites the actual fetched sources. Nothing is fabricated.

Run:  streamlit run 04_web_research_agent/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import run_web_research  # noqa: E402
from llm import get_secret, missing_key_message  # noqa: E402

st.set_page_config(page_title="Web Research Agent", page_icon="🌐", layout="wide")

STAGES = [
    ("search", "🔍 Search", "turn the question into queries and find result URLs"),
    ("retrieve", "📥 Retrieve", "fetch and clean the pages"),
    ("extract", "✂️ Extract", "pull the relevant passages verbatim"),
    ("analyze", "🧠 Analyze", "derive findings with source tags"),
    ("summarize", "📝 Summarize", "write the final cited answer"),
]


def main():
    st.title("🌐 Web Research Agent")
    st.caption("LangGraph · search → retrieve → extract → analyze → summarize · real sources only")

    if not get_secret("GROQ_API_KEY"):
        st.warning(missing_key_message())

    with st.sidebar:
        st.header("⚙️ Settings")
        max_sources = st.slider("Max pages to fetch", 2, 8, 4)
        st.divider()
        st.caption(f"LLM: {get_secret('GROQ_MODEL', 'openai/gpt-oss-120b')} (Groq)")
        st.caption("Search: DuckDuckGo / Bing (live)")
        st.divider()
        st.markdown("**Pipeline**")
        for key, label, desc in STAGES:
            st.markdown(f"{label} — _{desc}_")
        st.divider()
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("wr", None)
            st.rerun()

    question = st.text_area(
        "Research question",
        placeholder="e.g. What are the main causes of the 2024 CrowdStrike outage and how was it resolved?",
        height=90,
    )
    run = st.button("🌐 Research", type="primary", disabled=not question.strip())

    if run:
        with st.spinner("Searching, fetching and analysing live web sources…"):
            st.session_state["wr"] = run_web_research(question, max_sources=max_sources)

    res = st.session_state.get("wr")
    if not res:
        st.info("Ask a question to start the web research pipeline.")
        return

    if res.get("queries"):
        st.info("**Queries:** " + " · ".join(f"`{q}`" for q in res["queries"]))

    # Stage trace
    st.subheader("🧭 Research trail")
    for step in res.get("trace", []):
        label = {k: v[0] for k, v in STAGES and [(s[0], (s[1],)) for s in STAGES]}.get(step["agent"], step["agent"])
        st.markdown(f"{label} — {step['message']}")

    if res.get("error"):
        st.warning(f"Note: {res['error']}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("✅ Answer")
        st.markdown(res.get("answer") or "_(no answer)_")
    with col2:
        st.subheader(f"🔗 Sources ({len(res.get('sources', []))})")
        for s in res.get("sources", []):
            st.markdown(f"**[{s['tag']}]** [{s['title'] or s['url']}]({s['url']})")

    with st.expander(f"✂️ Extracted passages ({len(res.get('passages', []))})", expanded=False):
        for p in res.get("passages", []):
            st.markdown(f"**{p['title'] or p['url']}**\n\n> {p['quote']}")

    with st.expander(f"🧠 Findings ({len(res.get('findings', []))})", expanded=False):
        for f in res.get("findings", []):
            st.markdown(f"- {f}")

    with st.expander("📥 Export"):
        st.download_button(
            "Download markdown report",
            data=_markdown(question, res),
            file_name="web_research_report.md",
            mime="text/markdown",
        )


def _markdown(question: str, r: dict) -> str:
    out = [f"# Web research report\n\n**Question:** {question}\n", "## Answer\n", r.get("answer", ""), "\n## Sources\n"]
    for s in r.get("sources", []):
        out.append(f"- [{s['tag']}] {s['title']} — {s['url']}")
    if r.get("findings"):
        out.append("\n## Findings\n")
        out.extend(f"- {f}" for f in r["findings"])
    return "\n".join(out)


if __name__ == "__main__":
    main()
