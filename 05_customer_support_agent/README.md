# 🎧 Project 05 — Customer Support Agent

A **LangGraph** support agent that reads a customer message, classifies it,
retrieves the relevant help-centre articles, writes a grounded reply, has that
reply **reviewed by a QA pass**, and either finalises it or escalates to a human.
Policies are never invented — every claim traces back to a knowledge-base article.

## Workflow

```
Customer Message
      │
      ▼
Intent Classification ──► Knowledge Retrieval ──┬─(match)──► Response Generation
      │                                          └─(none)──► Escalate (human)
      │                                                          │
      └─────────────────► Review ◄──────────────────────────────┘
                            │
                   ┌────────┴─────────┐
              (approved)         (needs fix)
                   │                 │
                   ▼                 ▼
                Finalize  ◄────── Revise
                   │
                   ▼
              Final Answer + Sources
```

## Features

- **Intent + urgency classification** (billing, refund, technical_issue, …)
- **Real knowledge base**: 12 curated SaaS support articles across 8 categories
- **Hybrid retrieval**: keyword/tag scoring blended with semantic similarity
  (fastembed, optional) — no index build required
- **Grounded generation** with inline `[KB-xxx]` citations
- **QA reviewer pass** with a conditional edge (`finalize` vs `revise`)
- **Escalation path** when nothing relevant is retrieved (never fabricates)
- Streamlit UI with intent/urgency metrics, reply, reviewer note and sources

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (entry point) |
| `graph.py` | LangGraph workflow + `handle_ticket()` |
| `knowledge_base.py` | The curated support articles |
| `tools.py` | Hybrid retrieval + context formatting |
| `llm.py` | Groq LLM wiring + robust text extraction |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Streamlit runtime config |

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

| Setting | Value |
|---------|-------|
| Repository | `deepvisionkararhaider-crypto/langchain-langgraph-ai-lab` |
| Branch | `main` |
| Main file path | `05_customer_support_agent/app.py` |

**Required secrets:**

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## Notes / limitations

- The knowledge base is a self-contained demo corpus; swap `knowledge_base.py`
  for your own articles to productionise it.
- Escalation is automatic when retrieval finds no relevant article.

**Status:** ✅ Built · ✅ Tested locally · 🔜 Ready for Streamlit deployment
