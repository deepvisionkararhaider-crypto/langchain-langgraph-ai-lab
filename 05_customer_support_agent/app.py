"""Project 05 — Customer Support Agent (Streamlit entry point).

A LangGraph support pipeline:
  classify intent → retrieve KB → generate reply → review → (revise | finalize)
                                              └─(no KB match)─► escalate

Every reply is grounded in a curated knowledge base, checked by a QA reviewer
pass, and labelled with intent + urgency. Nothing is fabricated; when no article
matches, the ticket is escalated to a human instead.

Run:  streamlit run 05_customer_support_agent/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import handle_ticket  # noqa: E402
from knowledge_base import all_articles, categories  # noqa: E402
from llm import get_secret, missing_key_message  # noqa: E402

st.set_page_config(page_title="Customer Support Agent", page_icon="🎧", layout="wide")

URGENCY_COLOR = {"high": "🔴", "medium": "🟠", "low": "🟢"}

EXAMPLES = [
    "How do I upgrade from Starter to Pro, and does it cost more mid-cycle?",
    "I was charged twice for last month. Can I get a refund?",
    "My dashboard is showing a 500 error and won't load at all.",
    "How do I enable two-factor authentication?",
    "What is your uptime SLA on the Pro plan?",
    "Do you ship hardware internationally and how long does it take?",
]


def main():
    st.title("🎧 Customer Support Agent")
    st.caption("LangGraph · intent → KB retrieval → generate → review → finalize/escalate")

    if not get_secret("GROQ_API_KEY"):
        st.warning(missing_key_message())

    with st.sidebar:
        st.header("📚 Knowledge base")
        st.metric("Articles", len(all_articles()))
        st.caption("Categories: " + ", ".join(categories()))
        st.divider()
        st.header("💬 Example tickets")
        for ex in EXAMPLES:
            if st.button(ex, key=f"ex_{ex[:24]}", use_container_width=True):
                st.session_state["msg"] = ex
        st.divider()
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("ticket", None)
            st.session_state.pop("msg", None)
            st.rerun()

    message = st.text_area(
        "Customer message",
        value=st.session_state.get("msg", ""),
        placeholder="e.g. I need to add a VAT number to my invoice…",
        height=100,
    )
    run = st.button("📨 Handle ticket", type="primary", disabled=not message.strip())

    if run:
        with st.spinner("Classifying, retrieving and drafting a reply…"):
            st.session_state["ticket"] = handle_ticket(message)

    res = st.session_state.get("ticket")
    if not res:
        st.info("Paste a customer message or pick an example to see the agent work.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", res.get("intent", "other"))
    c2.metric("Urgency", f"{URGENCY_COLOR.get(res.get('urgency','medium'),'')} {res.get('urgency','medium')}")
    c3.metric("Escalated", "Yes" if res.get("escalate") else "No")

    st.subheader("✉️ Reply")
    st.markdown(res.get("answer") or "_(no reply)_")

    if res.get("escalate"):
        st.warning("No knowledge-base article matched — this ticket was routed to a human specialist.")
    else:
        st.success("Reply grounded in the knowledge base and approved by the reviewer.")

    if res.get("review"):
        st.caption(f"🕵️ Reviewer note: {res['review']}")

    if res.get("sources"):
        st.subheader("📚 Sources")
        for s in res["sources"]:
            st.markdown(f"- **[{s['id']}]** {s['title']} _(category: {s['category']}, score {s['score']})_")

    with st.expander("🧭 Agent steps"):
        for step in res.get("steps", []):
            st.markdown(f"- {step}")


if __name__ == "__main__":
    main()
