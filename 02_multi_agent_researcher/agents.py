"""Project 02 — Multi-Agent Researcher: agent definitions.

Four specialised LangGraph agents collaborate on a research question:

    ResearchAgent → plans sub-queries and gathers real web evidence
    FactAgent     → extracts factual, source-backed claims (no fabrication)
    AnalysisAgent → synthesises findings, notes agreements/conflicts/gaps
    WriterAgent   → writes the final cited answer

Each agent is a plain function over the shared ``ResearchState`` so the graph in
``graph.py`` stays declarative.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from llm import extract_text, get_llm
from tools import SearchHit, gather_evidence

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    question: str
    depth: int
    max_sources: int
    subqueries: list[str]
    evidence: list[dict]          # [{url,title,text}]
    facts: list[dict]             # [{claim, source}]
    analysis: str
    answer: str
    sources: list[dict]
    error: str
    trace: Annotated[list[dict], lambda a, b: (a or []) + (b or [])]


def _log(state: ResearchState, agent: str, message: str) -> list[dict]:
    return [{"agent": agent, "message": message}]


def _ctx(state: ResearchState, limit: int = 2800) -> str:
    chunks = []
    for i, e in enumerate(state.get("evidence", []), start=1):
        chunks.append(f"[S{i}] {e.get('title','')} — {e.get('url','')}\n{e.get('text','')[:limit]}")
    return "\n\n".join(chunks) if chunks else "(no evidence gathered)"


# ---------------------------------------------------------------------------
# Agent 1 — Research (plan + gather)
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are a research planner. Given a question, produce {n} focused
web-search queries that together answer it. Return ONE query per line, no numbering,
no extra text.

Question: {question}"""


def research_agent(state: ResearchState) -> dict:
    question = state["question"]
    depth = int(state.get("depth", 3))
    max_sources = int(state.get("max_sources", 4))
    subqueries: list[str] = [question]
    note = "used question as single query"
    try:
        llm = get_llm(temperature=0.2, max_tokens=512)
        raw = extract_text(llm.invoke(PLANNER_PROMPT.format(n=depth, question=question)))
        planned = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
        planned = [q for q in planned if len(q) > 4][:depth]
        if planned:
            subqueries = planned
            note = f"planned {len(planned)} sub-queries"
    except Exception as exc:  # planner is best-effort
        note = f"planner fallback ({type(exc).__name__})"

    evidence: list[dict] = []
    seen: set[str] = set()
    for q in subqueries:
        bundle = gather_evidence(q, max_sources=max_sources)
        for p in bundle["pages"]:
            if p.url in seen:
                continue
            seen.add(p.url)
            evidence.append({"url": p.url, "title": p.title or p.url, "text": p.text})

    return {
        "subqueries": subqueries,
        "evidence": evidence[: max_sources * 2],
        "trace": _log(state, "ResearchAgent", f"{note}; fetched {len(evidence)} page(s) after dedupe"),
    }


# ---------------------------------------------------------------------------
# Agent 2 — Fact extraction
# ---------------------------------------------------------------------------
FACT_PROMPT = """Extract the most important factual claims that help answer the
question. Use ONLY the evidence. For each claim give a source tag like [S1].

Question: {question}

Evidence:
{context}

Return up to 8 bullet lines formatted exactly as:
- <claim> [S<n>]"""


def fact_agent(state: ResearchState) -> dict:
    if not state.get("evidence"):
        return {"facts": [], "trace": _log(state, "FactAgent", "no evidence to extract from")}
    try:
        llm = get_llm(temperature=0.1, max_tokens=1024)
        raw = extract_text(llm.invoke(FACT_PROMPT.format(question=state["question"], context=_ctx(state))))
    except Exception as exc:
        return {"facts": [], "error": str(exc), "trace": _log(state, "FactAgent", f"failed: {exc}")}

    facts: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith(("-", "*", "•")):
            continue
        body = line.lstrip("-*• ").strip()
        if not body:
            continue
        import re

        m = re.search(r"\[S(\d+)\]", body)
        facts.append({"claim": re.sub(r"\[S\d+\]", "", body).strip(), "source": int(m.group(1)) if m else None})
    return {"facts": facts, "trace": _log(state, "FactAgent", f"extracted {len(facts)} claim(s)")}


# ---------------------------------------------------------------------------
# Agent 3 — Analysis / synthesis
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """You are an analyst. Using the evidence and extracted facts, write a
tight analytical brief that:
- answers the question directly,
- groups related findings,
- flags any conflicting or uncertain points and gaps,
- does NOT invent facts beyond the evidence.

Question: {question}

Extracted facts:
{facts}

Evidence:
{context}"""


def analysis_agent(state: ResearchState) -> dict:
    facts_txt = "\n".join(
        f"- {f['claim']}" + (f" [S{f['source']}]" if f.get("source") else "")
        for f in state.get("facts", [])
    ) or "(none extracted)"
    try:
        llm = get_llm(temperature=0.2, max_tokens=1200)
        analysis = extract_text(
            llm.invoke(ANALYSIS_PROMPT.format(question=state["question"], facts=facts_txt, context=_ctx(state)))
        )
    except Exception as exc:
        analysis = ""
        return {"analysis": analysis, "error": str(exc), "trace": _log(state, "AnalysisAgent", f"failed: {exc}")}
    return {"analysis": analysis, "trace": _log(state, "AnalysisAgent", "produced analysis brief")}


# ---------------------------------------------------------------------------
# Agent 4 — Writer
# ---------------------------------------------------------------------------
WRITER_PROMPT = """You are the final writer. Turn the analysis into a clear, well-structured
answer for the user. Rules:
- Ground every factual statement in the evidence; cite with [S1], [S2] …
- Start with a 2-3 sentence direct answer, then use short sections/bullets.
- If evidence is thin or conflicting, say so explicitly.
- Never fabricate sources or facts.

Question: {question}

Analysis:
{analysis}

Evidence (with source tags):
{context}"""


def writer_agent(state: ResearchState) -> dict:
    sources = [
        {"tag": f"S{i}", "title": e.get("title", ""), "url": e.get("url", "")}
        for i, e in enumerate(state.get("evidence", []), start=1)
    ]
    try:
        llm = get_llm(temperature=0.3, max_tokens=1500)
        answer = extract_text(
            llm.invoke(
                WRITER_PROMPT.format(
                    question=state["question"],
                    analysis=state.get("analysis") or "(no analysis)",
                    context=_ctx(state),
                )
            )
        )
    except Exception as exc:
        answer = f"⚠️ Could not write the final answer: {exc}"
    return {"answer": answer, "sources": sources, "trace": _log(state, "WriterAgent", f"wrote answer with {len(sources)} source(s)")}
