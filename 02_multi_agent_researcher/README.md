# 🔎 Project 02 — Multi-Agent Researcher

A **LangGraph** team of four specialised agents that researches a question using
**real web search + page extraction**, extracts source-backed facts, analyses
them, and writes a cited final answer. Sources are the actual fetched URLs —
nothing is fabricated.

## Agents

| Agent | Role |
|-------|------|
| 🔍 **ResearchAgent** | Plans focused sub-queries and gathers real web evidence (DDG search + page fetch) |
| 🧾 **FactAgent** | Extracts atomic factual claims, each tagged to its source `[S1]`, `[S2]` … |
| 🧠 **AnalysisAgent** | Synthesises findings, notes agreements, conflicts and gaps |
| ✍️ **WriterAgent** | Produces the final, structured, source-cited answer |

## Architecture

```
                 User Question
                       │
               ┌───────▼────────┐
               │ ResearchAgent  │  plan sub-queries → search → fetch pages
               └───────┬────────┘
               ┌───────▼────────┐
               │  FactAgent     │  grounded claims + source tags
               └───────┬────────┘
               ┌───────▼────────┐
               │ AnalysisAgent  │  synthesis / conflicts / gaps
               └───────┬────────┘
               ┌───────▼────────┐
               │  WriterAgent   │  cited final answer
               └───────┬────────┘
                       ▼
                 Final Answer + Sources
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (entry point) |
| `agents.py` | The four agent functions + shared `ResearchState` |
| `graph.py` | LangGraph wiring + `run_research()` |
| `tools.py` | Web search (ddgs → DDG HTML fallback) + page fetcher/extractor |
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
| Main file path | `02_multi_agent_researcher/app.py` |

**Required secrets:**

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## How to use

1. Type a research question.
2. (Optional) adjust research breadth and sources per sub-query in the sidebar.
3. Press **Run research** — watch the agent pipeline execute.
4. Read the final answer, then the extracted facts, analysis brief and sources.
5. Export a markdown report.

## Notes / limitations

- Live web access is required; if no sources are reachable the app says so
  instead of inventing an answer.
- Search quality depends on DuckDuckGo availability (two backends are tried).
- Only a handful of pages are fetched per run to keep latency low.

**Status:** ✅ Built · ✅ Tested locally · 🔜 Ready for Streamlit deployment
