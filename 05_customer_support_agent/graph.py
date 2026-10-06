"""Project 05 — Customer Support Agent: LangGraph workflow.

    START → classify_intent → retrieve_kb → generate → review ─┬─(approved)──► finalize → END
                                                               └─(needs fix)──► revise  → finalize
                                        (low confidence) ──────────────────────► escalate → END

The reviewer is a second LLM pass that checks the draft against the retrieved
knowledge base; a conditional edge either finalises it or sends it back for one
grounded revision. Escalation is used when nothing relevant was retrieved.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from knowledge_base import by_id
from llm import extract_text, get_llm
from tools import format_context, retrieve

INTENTS = [
    "billing", "refund", "technical_issue", "account_access", "plans_pricing",
    "shipping", "security", "integration", "how_to", "other",
]


class SupportState(TypedDict, total=False):
    question: str
    history: list[dict]
    intent: str
    urgency: str
    matches: list[tuple]         # (Article, score)
    context: str
    draft: str
    review: str
    approved: bool
    answer: str
    sources: list[dict]
    escalate: bool
    steps: Annotated[list[str], lambda a, b: (a or []) + (b or [])]


# ---------------------------------------------------------------------------
# 1. Intent + urgency
# ---------------------------------------------------------------------------
INTENT_PROMPT = """Classify the customer message. Reply in exactly two lines:
intent: one of [{intents}]
urgency: one of [low, medium, high]

Message: {question}"""


def classify_intent(state: SupportState) -> dict:
    question = state["question"]
    intent, urgency = "other", "medium"
    try:
        llm = get_llm(temperature=0.0, max_tokens=60)
        raw = extract_text(llm.invoke(INTENT_PROMPT.format(intents=", ".join(INTENTS), question=question)))
        for line in raw.splitlines():
            low = line.lower()
            if "intent" in low:
                for cand in INTENTS:
                    if cand in low:
                        intent = cand
                        break
            elif "urgency" in low:
                for u in ("high", "medium", "low"):
                    if u in low:
                        urgency = u
                        break
    except Exception:
        pass
    return {"intent": intent, "urgency": urgency, "steps": [f"classify: intent={intent}, urgency={urgency}"]}


# ---------------------------------------------------------------------------
# 2. Knowledge retrieval
# ---------------------------------------------------------------------------
def retrieve_kb(state: SupportState) -> dict:
    matches = retrieve(state["question"], k=3)
    return {
        "matches": matches,
        "context": format_context(matches) if matches else "(no relevant articles found)",
        "steps": [f"retrieve: {len(matches)} article(s)"],
    }


def route_after_retrieve(state: SupportState) -> str:
    return "generate" if state.get("matches") else "escalate"


# ---------------------------------------------------------------------------
# 3. Response generation
# ---------------------------------------------------------------------------
GEN_PROMPT = """You are a friendly, precise customer-support agent for a SaaS product.

Using ONLY the knowledge-base articles below, write a reply to the customer.
Rules:
- Be warm, concise and specific. Use short paragraphs or bullets.
- If steps are involved, give them as a numbered list.
- Cite articles inline like [KB-003].
- If the articles do not fully answer the question, say what is missing and offer
  to escalate — never invent policies, prices or timelines.
- Sign off as "AcmeCloud Support".

Customer message: {question}

Knowledge base:
{context}

Reply:"""


def generate(state: SupportState) -> dict:
    try:
        llm = get_llm(temperature=0.3, max_tokens=900)
        draft = extract_text(llm.invoke(GEN_PROMPT.format(
            question=state["question"], context=state.get("context", "")
        )))
    except Exception as exc:
        draft = f"(generation failed: {exc})"
    return {"draft": draft, "steps": ["generate: draft reply written"]}


# ---------------------------------------------------------------------------
# 4. Review (grounding check)
# ---------------------------------------------------------------------------
REVIEW_PROMPT = """You are a support QA reviewer. Check whether the draft reply is fully
supported by the knowledge base and whether it answers the customer.

Return exactly:
verdict: APPROVED or REVISE
reason: one short sentence

If REVISE, also return a corrected reply after a line 'REVISED:' that removes any
unsupported claims and stays grounded in the knowledge base.

Customer message: {question}

Knowledge base:
{context}

Draft reply:
{draft}"""


def review(state: SupportState) -> dict:
    try:
        llm = get_llm(temperature=0.0, max_tokens=1100)
        raw = extract_text(llm.invoke(REVIEW_PROMPT.format(
            question=state["question"], context=state.get("context", ""), draft=state.get("draft", "")
        )))
    except Exception as exc:
        return {"approved": True, "review": f"(reviewer unavailable: {exc})", "draft": state.get("draft", ""),
                "steps": ["review: skipped (error)"]}

    approved = "APPROVED" in raw.upper().split("REVISED:")[0]
    reason = ""
    for line in raw.splitlines():
        if line.lower().startswith("reason"):
            reason = line.split(":", 1)[-1].strip()
            break
    revised = ""
    if "REVISED:" in raw:
        revised = raw.split("REVISED:", 1)[1].strip()
    return {
        "approved": approved,
        "review": reason or raw[:200],
        "draft": revised or state.get("draft", ""),
        "steps": [f"review: {'approved' if approved else 'revised'} ({reason or 'ok'})"],
    }


def route_after_review(state: SupportState) -> str:
    return "finalize" if state.get("approved") else "revise"


# ---------------------------------------------------------------------------
# 5. Revise (one grounded pass) & finalize / escalate
# ---------------------------------------------------------------------------
def revise(state: SupportState) -> dict:
    if state.get("review") and state.get("draft"):
        # review() already substituted a corrected draft when available
        return {"approved": True, "steps": ["revise: corrected draft accepted"]}
    return {"steps": ["revise: no change"]}


def _collect_sources(state: SupportState) -> list[dict]:
    out = []
    for a, score in state.get("matches", []):
        out.append({"id": a.id, "title": a.title, "category": a.category, "score": score})
    return out


def finalize(state: SupportState) -> dict:
    return {
        "answer": state.get("draft", ""),
        "sources": _collect_sources(state),
        "escalate": False,
        "steps": ["finalize: reply ready"],
    }


ESCALATE_PROMPT = """No knowledge-base article matched this customer message. Write a short,
honest reply that: acknowledges the request, explains you don't have a confident
answer, and offers to escalate to a human specialist (P1/P2 if urgent). Do not
invent policies.

Customer message: {question}"""


def escalate(state: SupportState) -> dict:
    try:
        llm = get_llm(temperature=0.2, max_tokens=400)
        answer = extract_text(llm.invoke(ESCALATE_PROMPT.format(question=state["question"])))
    except Exception as exc:
        answer = f"(escalation message failed: {exc})"
    return {
        "answer": answer,
        "sources": [],
        "escalate": True,
        "steps": ["escalate: no KB match — routed to human"],
    }


def build_graph():
    g = StateGraph(SupportState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_kb", retrieve_kb)
    g.add_node("generate", generate)
    g.add_node("review", review)
    g.add_node("revise", revise)
    g.add_node("finalize", finalize)
    g.add_node("escalate", escalate)

    g.add_edge(START, "classify_intent")
    g.add_edge("classify_intent", "retrieve_kb")
    g.add_conditional_edges("retrieve_kb", route_after_retrieve, {"generate": "generate", "escalate": "escalate"})
    g.add_edge("generate", "review")
    g.add_conditional_edges("review", route_after_review, {"finalize": "finalize", "revise": "revise"})
    g.add_edge("revise", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g.compile()


def handle_ticket(question: str) -> dict[str, Any]:
    graph = build_graph()
    out = graph.invoke({"question": question})
    return {
        "answer": out.get("answer", ""),
        "intent": out.get("intent", "other"),
        "urgency": out.get("urgency", "medium"),
        "sources": out.get("sources", []),
        "escalate": out.get("escalate", False),
        "review": out.get("review", ""),
        "steps": out.get("steps", []),
    }
