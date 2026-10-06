"""Project 02 — Multi-Agent Researcher: LangGraph orchestration.

    START → research → facts → analysis → writer → END

Linear hand-off between four specialised agents, each contributing to the shared
``ResearchState``. Every node appends to ``trace`` so the UI can show the agent
pipeline as it runs.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents import (
    ResearchState,
    analysis_agent,
    fact_agent,
    research_agent,
    writer_agent,
)


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("research", research_agent)
    g.add_node("facts", fact_agent)
    g.add_node("analysis", analysis_agent)
    g.add_node("writer", writer_agent)
    g.add_edge(START, "research")
    g.add_edge("research", "facts")
    g.add_edge("facts", "analysis")
    g.add_edge("analysis", "writer")
    g.add_edge("writer", END)
    return g.compile()


def run_research(question: str, depth: int = 3, max_sources: int = 4) -> dict[str, Any]:
    """Execute the full multi-agent pipeline for a question."""
    graph = build_graph()
    state: ResearchState = {
        "question": question,
        "depth": depth,
        "max_sources": max_sources,
    }
    result = graph.invoke(state)
    return {
        "answer": result.get("answer", ""),
        "subqueries": result.get("subqueries", []),
        "facts": result.get("facts", []),
        "analysis": result.get("analysis", ""),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "error": result.get("error"),
    }
