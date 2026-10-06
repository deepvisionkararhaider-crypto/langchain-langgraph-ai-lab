"""Project 04 — Web Research Agent: LangGraph search→retrieve→extract→analyze→summarize.

    START → search → retrieve → extract → analyze → summarize → END

Distinct responsibilities per node:
  search    — turn the question into queries and find result URLs
  retrieve  — fetch + clean the pages behind those URLs
  extract   — pull the most relevant passages from each page (no fabrication)
  analyze   — derive findings, with source tags [S1..Sn]
  summarize — write the final answer grounded only in the extracted evidence
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import extract_text, get_llm
from tools import fetch_page, web_search


class WebResearchState(TypedDict, total=False):
    question: str
    max_sources: int
    queries: list[str]
    hits: list[dict]          # {title,url,snippet}
    pages: list[dict]         # {url,title,text}
    passages: list[dict]      # {url,title,quote}
    findings: list[str]
    answer: str
    sources: list[dict]
    error: str
    trace: Annotated[list[dict], lambda a, b: (a or []) + (b or [])]


def _log(agent: str, message: str) -> list[dict]:
    return [{"agent": agent, "message": message}]


# ---------------------------------------------------------------------------
# 1. Search
# ---------------------------------------------------------------------------
QUERY_PROMPT = """Write up to 3 short, high-signal web search queries that together
answer the question. One per line, no numbering, no quotes, no extra text.

Question: {question}"""


def search_node(state: WebResearchState) -> dict:
    q = state["question"]
    queries = [q]
    try:
        llm = get_llm(temperature=0.2, max_tokens=256)
        raw = extract_text(llm.invoke(QUERY_PROMPT.format(question=q)))
        planned = [ln.strip(" -•\t\"'") for ln in raw.splitlines() if len(ln.strip()) > 4][:3]
        if planned:
            queries = planned
    except Exception:
        pass
    hits: list[dict] = []
    seen: set[str] = set()
    for query in queries:
        for h in web_search(query, max_results=6):
            if h.url not in seen:
                seen.add(h.url)
                hits.append({"title": h.title, "url": h.url, "snippet": h.snippet})
    return {
        "queries": queries,
        "hits": hits,
        "trace": _log("search", f"{len(queries)} quer(y/ies) → {len(hits)} unique result(s)"),
    }


# ---------------------------------------------------------------------------
# 2. Retrieve
# ---------------------------------------------------------------------------
def retrieve_node(state: WebResearchState) -> dict:
    limit = int(state.get("max_sources", 4))
    pages: list[dict] = []
    for h in state.get("hits", []):
        if len(pages) >= limit:
            break
        page = fetch_page(h["url"], max_chars=4000)
        if page.ok and len(page.text) > 200:
            pages.append({"url": page.url, "title": page.title or h["title"], "text": page.text})
    return {"pages": pages, "trace": _log("retrieve", f"fetched {len(pages)} page(s)")}


def route_after_retrieve(state: WebResearchState) -> str:
    return "extract" if state.get("pages") else "summarize"


# ---------------------------------------------------------------------------
# 3. Extract relevant passages
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """From the page text below, extract 1-3 verbatim passages that
directly help answer the question. Each passage on its own line prefixed with "- ".
Do not paraphrase, do not add commentary, do not invent text. If nothing is relevant,
output exactly: NONE

Question: {question}

Page: {title} ({url})
Text:
{text}"""


def extract_node(state: WebResearchState) -> dict:
    passages: list[dict] = []
    llm = get_llm(temperature=0.0, max_tokens=700)
    for i, p in enumerate(state.get("pages", []), start=1):
        try:
            raw = extract_text(
                llm.invoke(EXTRACT_PROMPT.format(
                    question=state["question"], title=p["title"], url=p["url"], text=p["text"]
                ))
            )
        except Exception:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith(("-", "*", "•")):
                quote = line.lstrip("-*• ").strip()
                if quote and quote.upper() != "NONE" and len(quote) > 20:
                    passages.append({"url": p["url"], "title": p["title"], "quote": quote})
    return {"passages": passages, "trace": _log("extract", f"{len(passages)} relevant passage(s)")}


# ---------------------------------------------------------------------------
# 4. Analyze
# ---------------------------------------------------------------------------
def _evidence_block(state: WebResearchState) -> str:
    by_url: dict[str, list[str]] = {}
    for p in state.get("passages", []):
        by_url.setdefault(p["url"], []).append(p["quote"])
    tag = {u: f"S{i}" for i, (u, _) in enumerate(by_url.items(), start=1)}
    lines = []
    for url, quotes in by_url.items():
        lines.append(f"[{tag[url]}] {url}")
        lines.extend(f"  - {q}" for q in quotes)
    return "\n".join(lines) if lines else "(no evidence)"


ANALYZE_PROMPT = """You are an analyst. Using ONLY the numbered evidence below, list the key
findings that answer the question. Each finding must cite its source tag(s) like [S1].
Note contradictions or gaps. Do not invent facts. 4-8 bullet lines.

Question: {question}

Evidence:
{evidence}"""


def analyze_node(state: WebResearchState) -> dict:
    try:
        llm = get_llm(temperature=0.2, max_tokens=900)
        raw = extract_text(llm.invoke(ANALYZE_PROMPT.format(
            question=state["question"], evidence=_evidence_block(state)
        )))
        findings = [ln.strip().lstrip("-*• ").strip() for ln in raw.splitlines() if ln.strip().startswith(("-", "*", "•"))]
    except Exception as exc:
        findings = []
        return {"findings": findings, "error": str(exc), "trace": _log("analyze", f"failed: {exc}")}
    return {"findings": findings, "trace": _log("analyze", f"{len(findings)} finding(s)")}


# ---------------------------------------------------------------------------
# 5. Summarize
# ---------------------------------------------------------------------------
SUMMARY_PROMPT = """Write the final answer to the user's question, grounded ONLY in the
evidence and findings. Cite sources as [S1], [S2] … Start with a 2-3 sentence direct
answer, then short bulleted details. If evidence is thin or conflicting, say so.
Never fabricate facts or sources.

Question: {question}

Findings:
{findings}

Evidence:
{evidence}"""


def summarize_node(state: WebResearchState) -> dict:
    sources: list[dict] = []
    seen: set[str] = set()
    for p in state.get("pages", []):
        if p["url"] not in seen:
            seen.add(p["url"])
            sources.append({"tag": f"S{len(sources)+1}", "title": p["title"], "url": p["url"]})
    if not state.get("pages"):
        return {
            "answer": "I could not reach any web sources from this environment, so I have nothing grounded to report.",
            "sources": [],
            "trace": _log("summarize", "no evidence — refused to fabricate"),
        }
    try:
        llm = get_llm(temperature=0.3, max_tokens=1300)
        answer = extract_text(llm.invoke(SUMMARY_PROMPT.format(
            question=state["question"],
            findings="\n".join(f"- {f}" for f in state.get("findings", [])) or "(none)",
            evidence=_evidence_block(state),
        )))
    except Exception as exc:
        answer = f"⚠️ Summarisation failed: {exc}"
    return {"answer": answer, "sources": sources, "trace": _log("summarize", f"answer written with {len(sources)} source(s)")}


def build_graph():
    g = StateGraph(WebResearchState)
    g.add_node("search", search_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("extract", extract_node)
    g.add_node("analyze", analyze_node)
    g.add_node("summarize", summarize_node)
    g.add_edge(START, "search")
    g.add_edge("search", "retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"extract": "extract", "summarize": "summarize"})
    g.add_edge("extract", "analyze")
    g.add_edge("analyze", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def run_web_research(question: str, max_sources: int = 4) -> dict[str, Any]:
    graph = build_graph()
    out = graph.invoke({"question": question, "max_sources": max_sources})
    return {
        "answer": out.get("answer", ""),
        "queries": out.get("queries", []),
        "passages": out.get("passages", []),
        "findings": out.get("findings", []),
        "sources": out.get("sources", []),
        "trace": out.get("trace", []),
        "error": out.get("error"),
    }
