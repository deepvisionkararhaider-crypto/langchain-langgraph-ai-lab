# 🧪 LangChain + LangGraph AI Lab

A portfolio of **10 real AI projects** in a single repository. Every project is a
self-contained **Streamlit** application powered by **LangChain** and **LangGraph**,
designed to be deployed on **Streamlit Community Cloud**.

> **Status:** Projects **01–05 are fully built, tested locally, committed, pushed and
> verified on GitHub.** Projects 06–10 are planned (see roadmap). No public Streamlit
> URLs are claimed — Streamlit Community Cloud requires interactive sign-in that
> cannot be performed from this environment.

---

## Portfolio overview

| # | Project | LangChain | LangGraph | Streamlit | GitHub | Live Demo |
|---|---------|:---------:|:---------:|:---------:|:------:|-----------|
| 1 | RAG Document Chat | ✅ | ✅ | ✅ | ✅ | https://blgzq8lsxwwhy2i4sotj2c.streamlit.app/ |
| 2 | Multi-Agent Researcher | ✅ | ✅ | ✅ | ✅ | https://ifrqkxze9s9f2eshvc8t6m.streamlit.app/ |
| 3 | SQL Agent | ✅ | ✅ | ✅ | ✅ | https://gnmadwnsqreoyqtxrmb4mv.streamlit.app/ |
| 4 | Web Research Agent | ✅ | ✅ | ✅ | ✅ | https://langchain-langgraph-ai-lab-acevfwnkkbkxtgeyehs6tk.streamlit.app/ |
| 5 | Customer Support Agent | ✅ | ✅ | ✅ | ✅ | https://v7jmuvfpvdkyikqqwkjrcj.streamlit.app/ |
| 6 | Document Extraction | 🚧 | 🚧 | 🚧 | 🚧 | Planned |
| 7 | Planner Executor | 🚧 | 🚧 | 🚧 | 🚧 | Planned |
| 8 | Code Review | 🚧 | 🚧 | 🚧 | 🚧 | Planned |
| 9 | Content Research | 🚧 | 🚧 | 🚧 | 🚧 | Planned |
| 10 | Memory Assistant | 🚧 | 🚧 | 🚧 | 🚧 | Planned |

**"Ready for Streamlit deployment"** means the app builds and runs locally, the
entry point and `requirements.txt` are in place, and the exact deploy settings are
documented — but it has **not** been published (that needs interactive auth I can't
perform). No URL is marked "Verified" because none has been verified live.

---

## Repository structure

```
langchain-langgraph-ai-lab/
├── app.py                      # Root dashboard (launcher; does not load AI models)
├── README.md
├── .gitignore
├── 01_rag_document_chat/       app.py rag.py graph.py llm.py README.md requirements.txt .streamlit/
├── 02_multi_agent_researcher/  app.py agents.py graph.py tools.py llm.py README.md requirements.txt .streamlit/
├── 03_sql_agent/               app.py agent.py graph.py database.py llm.py README.md requirements.txt .streamlit/
├── 04_web_research_agent/      app.py graph.py tools.py llm.py README.md requirements.txt .streamlit/
└── 05_customer_support_agent/  app.py graph.py tools.py knowledge_base.py llm.py README.md requirements.txt .streamlit/
```

Each project folder is **independently deployable**: its own `app.py` entry point,
its own `requirements.txt`, and its own `.streamlit/config.toml`.

---

## Architecture

```
                       ┌──────────────────────────────────────┐
                       │            Streamlit UI              │
                       │      (one app.py per project)        │
                       └──────────────────┬───────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │         LangGraph workflow           │
                       │  state · nodes · conditional edges   │
                       └──────────────────┬───────────────────┘
                                          │
        ┌─────────────────┬───────────────┼──────────────────┬─────────────────┐
        ▼                 ▼               ▼                  ▼                 ▼
   LangChain        Retrieval /      SQL / tools        Web search        Knowledge
   prompts          embeddings       (SQLAlchemy)       (ddgs/Bing)       base (hybrid)
        │                 │               │                  │                 │
        └─────────────────┴───────────────┴──────────────────┴─────────────────┘
                                          │
                              ┌───────────▼────────────┐
                              │  LLM: Groq (gpt-oss)   │
                              └────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/deepvisionkararhaider-crypto/langchain-langgraph-ai-lab.git
cd langchain-langgraph-ai-lab

# pick a project
cd 01_rag_document_chat
pip install -r requirements.txt

export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

Root dashboard:

```bash
pip install streamlit
streamlit run app.py
```

---

## Environment variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key (https://console.groq.com/keys) |
| `GROQ_MODEL` | ➖ | Override the model (default `openai/gpt-oss-120b`) |

Provide them either as environment variables **or** via Streamlit secrets
(`.streamlit/secrets.toml`, or the *Secrets* box on Streamlit Cloud).
A template is provided at `**/.streamlit/secrets.toml.example`.
**Never commit real keys** — `.gitignore` excludes `.env` and `secrets.toml`.

---

## Project list (built & verified)

### 01 · 📄 RAG Document Chat
Upload PDF/DOCX/TXT/MD → load → chunk → embed (fastembed/ONNX) → FAISS → retrieve
→ LangGraph → grounded answer with source references.
Graph: `retrieve → route → generate`.

### 02 · 🔎 Multi-Agent Researcher
Four collaborating agents gather **real** web evidence and write a cited answer.
Graph: `research → facts → analysis → writer`.

### 03 · 🗄️ SQL Database Agent
Real text-to-SQL over SQLite with schema introspection, destructive-statement
guardrails and plain-language explanations.
Graph: `introspect → generate → guard → execute → explain`.

### 04 · 🌐 Web Research Agent
Live web research: fetch pages, extract verbatim passages, analyse, summarise.
Graph: `search → retrieve → extract → analyze → summarize`.

### 05 · 🎧 Customer Support Agent
Intent classification → KB retrieval → grounded reply → QA review → finalize/escalate.
Graph: `intent → retrieve → generate → review → finalize`.

---

## Roadmap (06–10)

| # | Project | Intended workflow |
|---|---------|-------------------|
| 6 | Document Extraction Pipeline | parse → extract → validate → structured output → summary (+ JSON download) |
| 7 | Planner + Executor Agent | planner → executor → reviewer → **conditional routing** → revision → final |
| 8 | Code Review Agent | static analysis → bugs → security → quality → reviewer → report (+ improved code) |
| 9 | Content Research Pipeline | topic → research → outline → draft → review → improve (length/tone/format) |
| 10 | Memory Personal Assistant | memory retrieval → agent → tool selection → response → memory update (persistent store) |

---

## LangChain concepts used

- **Chat models** via `langchain-groq` (`ChatGroq`)
- **Prompt templating** and message extraction
- **Document loaders**: `pypdf`, `python-docx`, text
- **Text splitters**: `RecursiveCharacterTextSplitter`
- **Embeddings + vector store**: `fastembed` + FAISS
- **Retrieval**: similarity retriever (`as_retriever`)
- **SQL toolkit**: `SQLDatabase` schema introspection
- **Utilities/tools**: web search + page extraction

## LangGraph concepts used

- **`StateGraph`** with a typed shared **state** (`TypedDict`)
- **Nodes** as pure functions over state; **reducers** for accumulating fields (e.g. `trace`, `steps`)
- **Conditional edges** for routing (safe/unsafe SQL, no-match escalation, review verdicts)
- **Compiled graphs** invoked per request; multi-agent hand-off
- **Review/fix loops** (generate → review → revise → finalize)

---

## Streamlit deployment

Every project deploys independently. In **Streamlit Community Cloud → New app**:

| Field | Value |
|-------|-------|
| Repository | `deepvisionkararhaider-crypto/langchain-langgraph-ai-lab` |
| Branch | `main` |
| Main file path | `<project_folder>/app.py` (e.g. `01_rag_document_chat/app.py`) |
| Secrets | `GROQ_API_KEY = "gsk_..."` |

Deploy paths:

```
01_rag_document_chat/app.py
02_multi_agent_researcher/app.py
03_sql_agent/app.py
04_web_research_agent/app.py
05_customer_support_agent/app.py
app.py                          # root dashboard
```

---

## GitHub structure

One repository, one folder per project, one commit per project:

```
feat: add project 01 rag document chat
feat: add project 02 multi agent researcher
feat: add project 03 sql agent
feat: add project 04 web research agent
feat: add project 05 customer support agent
```

---

## Notes & limitations

- **No public URLs are claimed.** Streamlit Community Cloud publishing requires
  interactive authorization that cannot be completed from the build environment;
  each project is instead marked *"Ready for Streamlit deployment"*.
- LLM output can vary; agents are prompted to stay grounded and are instructed to
  refuse rather than fabricate when evidence is missing.
- Web-research projects need outbound network access; if sources are unreachable
  the agent says so instead of inventing an answer.
- Each app keeps state in the Streamlit session (no shared server-side state).
