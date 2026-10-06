"""Project 01 — LangGraph RAG workflow.

Graph shape::

    [retrieve] ─► [generate] ─► END
         │
         └─(no context)─► [no_context] ─► END

State carries the question, the retrieved context, the formatted sources and
the final answer. The LLM only ever sees context returned by the retriever, so
answers are grounded — never hardcoded.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from llm import extract_text, get_llm

RAG_PROMPT = """You are a precise document question-answering assistant.

Answer the user's question using ONLY the context below. If the answer is not
contained in the context, reply exactly:
"I could not find that in the provided document."

Guidelines:
- Be concise and factual. Quote short spans when useful.
- Cite sources inline like [source: file, chunk N] where possible.
- Never invent information that is not in the context.

Context:
{context}

Question: {question}

Answer:"""


class RAGState(TypedDict, total=False):
    question: str
    k: int
    temperature: float
    documents: list[Document]
    context: str
    sources: list[dict]
    answer: str
    error: str
    steps: Annotated[list[str], lambda a, b: (a or []) + (b or [])]


def build_graph(retriever):
    """Compile the RAG LangGraph bound to a retriever."""

    def retrieve(state: RAGState) -> dict:
        question = state.get("question", "").strip()
        k = int(state.get("k", 4))
        if not question:
            return {"documents": [], "steps": ["retrieve: empty question"]}
        try:
            docs = retriever.invoke(question)[:k]
            return {"documents": docs, "steps": [f"retrieve: {len(docs)} chunk(s)"]}
        except Exception as exc:  # pragma: no cover - defensive
            return {"documents": [], "error": f"retrieval failed: {exc}", "steps": ["retrieve: error"]}

    def route(state: RAGState) -> str:
        return "generate" if state.get("documents") else "no_context"

    def generate(state: RAGState) -> dict:
        docs = state.get("documents", [])
        context = "\n\n---\n\n".join(
            f"[{ (d.metadata or {}).get('source','doc') } | chunk {(d.metadata or {}).get('chunk','?')}]\n{d.page_content}"
            for d in docs
        )
        try:
            llm = get_llm(temperature=float(state.get("temperature", 0.2)), max_tokens=1024)
            msg = llm.invoke(
                RAG_PROMPT.format(context=context, question=state.get("question", ""))
            )
            answer = extract_text(msg).strip() or "(empty model response)"
        except Exception as exc:
            answer = f"⚠️ Generation failed: {exc}"
        return {"context": context, "answer": answer, "steps": ["generate: answer produced"]}

    def no_context(state: RAGState) -> dict:
        return {
            "answer": "I could not find that in the provided document.",
            "steps": ["no_context: nothing retrieved"],
        }

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("no_context", no_context)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, {"generate": "generate", "no_context": "no_context"})
    g.add_edge("generate", END)
    g.add_edge("no_context", END)
    return g.compile()


def answer_question(retriever, question: str, k: int = 4, temperature: float = 0.2) -> dict:
    """Convenience wrapper used by the Streamlit app."""
    from rag import format_sources

    graph = build_graph(retriever)
    result: dict[str, Any] = graph.invoke(
        {"question": question, "k": k, "temperature": temperature}
    )
    docs = result.get("documents", []) or []
    return {
        "answer": result.get("answer", ""),
        "sources": format_sources(docs),
        "steps": result.get("steps", []),
        "error": result.get("error"),
    }
