# 🌐 Project 04 — Web Research Agent

A **LangGraph** web-research pipeline that answers a question from **live web
sources**. Each stage has one job, and the final answer cites the pages that
were actually fetched — sources are never fabricated.

## Workflow

```
Question ─► Search ─► Retrieve ─► Extract ─► Analyze ─► Summarize ─► Answer + Sources
             │           │           │           │           │
         queries     clean pages   verbatim    findings    cited
         + URLs      (fetched)     passages    [S1..Sn]    answer
                  └──(no pages)────────────────────► Summarize (refuse to fabricate)
```

## Stage responsibilities

| Stage | What it does |
|-------|--------------|
| 🔍 **Search** | LLM writes up to 3 focused queries; results collected & de-duplicated |
| 📥 **Retrieve** | Pages fetched (redirects followed), scripts/styles stripped, text capped |
| ✂️ **Extract** | Verbatim relevant passages pulled per page (no paraphrase) |
| 🧠 **Analyze** | Findings derived with `[S1]`, `[S2]` source tags; gaps noted |
| 📝 **Summarize** | Final cited answer; refuses to invent when evidence is missing |

Conditional edge: if **no pages** could be fetched, the graph skips extraction and
routes straight to a refusal summary.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (entry point) |
| `graph.py` | LangGraph nodes, conditional routing, `run_web_research()` |
| `tools.py` | Multi-backend live search (ddgs/DDG-HTML/Bing) + page fetcher |
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
| Main file path | `04_web_research_agent/app.py` |

**Required secrets:**

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## How to use

1. Enter a research question.
2. Adjust how many pages to fetch in the sidebar (2–8).
3. Press **Research** and watch the five-stage trail.
4. Inspect extracted passages, findings and sources; export a markdown report.

## Notes / limitations

- Requires outbound network access; if no source is reachable the agent says so
  rather than inventing an answer.
- Search relies on public endpoints (DuckDuckGo / Bing) which can rate-limit;
  three backends are tried in order.

**Status:** ✅ Built · ✅ Tested locally · 🔜 Ready for Streamlit deployment
