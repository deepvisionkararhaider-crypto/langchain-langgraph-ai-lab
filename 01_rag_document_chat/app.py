"""Project 01 — RAG Document Chat (Streamlit entry point).

Upload a PDF / DOCX / TXT / MD document, index it, then ask questions.
Answers are grounded in retrieved chunks; sources are shown for every answer.

Run:  streamlit run 01_rag_document_chat/app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

# Make sibling modules importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import (
    build_retriever,
    embedder_name,
    format_sources,
    ingest,
)  # noqa: E402
from graph import answer_question  # noqa: E402
from llm import get_secret, missing_key_message  # noqa: E402

st.set_page_config(page_title="RAG Document Chat", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def _embedder_label() -> str:
    return embedder_name()


def _sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 25)
        st.divider()
        st.caption(f"Embeddings: {_embedder_label()}")
        st.caption(f"LLM: {get_secret('GROQ_MODEL', 'openai/gpt-oss-120b')} (Groq)")
        if st.button("🗑️ Reset session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return k, temperature, chunk_size, chunk_overlap


def _ingest_uploads(files, chunk_size, chunk_overlap):
    all_chunks = []
    details = []
    for f in files:
        try:
            res = ingest(f.name, f.getvalue(), chunk_size, chunk_overlap)
            all_chunks.extend(res.chunks)
            details.append((res.source, res.n_chars, res.n_chunks))
        except Exception as exc:
            st.error(f"Failed to process **{f.name}**: {exc}")
    st.session_state["chunks"] = all_chunks
    st.session_state["ingest_details"] = details
    st.session_state["retriever"] = build_retriever(all_chunks, k=4) if all_chunks else None
    return all_chunks, details


def main():
    st.title("📄 RAG Document Chat")
    st.caption(
        "LangChain + LangGraph retrieval-augmented generation · "
        "upload → chunk → embed → retrieve → generate"
    )

    if not get_secret("GROQ_API_KEY"):
        st.warning(missing_key_message())

    k, temperature, chunk_size, chunk_overlap = _sidebar()

    # --- Upload / ingest -----------------------------------------------------
    with st.container(border=True):
        st.subheader("1 · Upload documents")
        files = st.file_uploader(
            "PDF, DOCX, TXT or Markdown",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("📥 Index documents", type="primary", disabled=not files):
            with st.spinner("Loading, splitting and embedding…"):
                chunks, details = _ingest_uploads(files, chunk_size, chunk_overlap)
            if chunks:
                st.success(f"Indexed {len(chunks)} chunks from {len(details)} file(s).")
            else:
                st.warning("No usable text found in the uploaded file(s).")

    details = st.session_state.get("ingest_details")
    if details:
        with st.expander(f"📊 Indexed {len(st.session_state.get('chunks', []))} chunks", expanded=False):
            st.dataframe(
                [{"file": d[0], "characters": d[1], "chunks": d[2]} for d in details],
                use_container_width=True,
            )

    st.divider()

    # --- Chat ----------------------------------------------------------------
    st.subheader("2 · Ask questions")
    history = st.session_state.setdefault("chat", [])
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["a"])
            if turn.get("sources"):
                with st.expander(f"🔎 Sources ({len(turn['sources'])})"):
                    for s in turn["sources"]:
                        st.markdown(
                            f"**{s['source']}** · chunk {s.get('chunk')} "
                            f"· rank {s['rank']}\n\n> {s['snippet']}…"
                        )

    question = st.chat_input("Ask something about your document…")
    if question:
        retriever = st.session_state.get("retriever")
        if retriever is None:
            st.info("Upload and index a document first.")
            return
        history.append({"q": question, "a": "", "sources": []})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                out = answer_question(retriever, question, k=k, temperature=temperature)
            st.markdown(out["answer"])
            if out.get("sources"):
                with st.expander(f"🔎 Sources ({len(out['sources'])})"):
                    for s in out["sources"]:
                        st.markdown(
                            f"**{s['source']}** · chunk {s.get('chunk')} "
                            f"· rank {s['rank']}\n\n> {s['snippet']}…"
                        )
            if out.get("steps"):
                st.caption("LangGraph: " + " → ".join(out["steps"]))
            history[-1]["a"] = out["answer"]
            history[-1]["sources"] = out.get("sources", [])


if __name__ == "__main__":
    main()
