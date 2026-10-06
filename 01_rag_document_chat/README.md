# 📄 Project 01 — RAG Document Chat

Conversational **Retrieval-Augmented Generation** over your own documents.
Upload a file, it gets chunked, embedded and indexed, then you ask questions —
every answer is grounded in retrieved passages with visible source references.

## Features

- **Upload**: PDF (`pypdf`), DOCX (`python-docx`), TXT, Markdown, CSV, JSON
- **Parsing → Chunking**: `RecursiveCharacterTextSplitter` (configurable size/overlap)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (offline, free)
- **Vector store**: FAISS (in-memory fallback if FAISS is unavailable)
- **Retriever**: similarity search with adjustable `k`
- **Orchestration**: **LangGraph** state machine (`retrieve → route → generate`)
- **LLM**: Groq `openai/gpt-oss-120b` via `langchain-groq`
- **Source references**: file + chunk id + snippet for every answer
- **No hardcoded answers**: the model only sees retrieved context; if nothing
  relevant is found it says so instead of inventing.

## Architecture

```
Document ─► Loader ─► Text Splitter ─► Embeddings ─► Vector Store
                                                          │
                                            Retriever ◄────┘
                                                │
                                        ┌───────▼────────┐
                                        │   LangGraph    │
                                        │  retrieve      │
                                        │     │          │
                                        │  route (cond.) │
                                        │   ├─► generate │
                                        │   └─► no_ctx   │
                                        └───────┬────────┘
                                                ▼
                                          Answer + Sources
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (entry point) |
| `rag.py` | Loaders, chunking, embeddings, vector store, retriever |
| `graph.py` | LangGraph RAG workflow |
| `llm.py` | Groq LLM wiring + robust text extraction |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Streamlit runtime config |

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."        # or use .streamlit/secrets.toml
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

| Setting | Value |
|---------|-------|
| Repository | `deepvisionkararhaider-crypto/langchain-langgraph-ai-lab` |
| Branch | `main` |
| Main file path | `01_rag_document_chat/app.py` |
| Requirements | auto-detected from `01_rag_document_chat/requirements.txt` |

**Required secrets** (paste into the app's *Secrets* box):

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

## How to use

1. Upload one or more documents in the **Upload documents** panel.
2. Click **Index documents** (watch the chunk count appear).
3. Ask questions in the chat box.
4. Expand **Sources** under any answer to see which chunks were used.
5. Tune `k`, temperature and chunking from the sidebar.

## Notes / limitations

- The index lives in the Streamlit session — reloading the page clears it.
- Scanned/image-only PDFs have no extractable text and will be reported as empty.
- `sentence-transformers` is downloaded on first run (~90 MB); the app falls
  back to a deterministic embedder if the download fails, so it never crashes.

**Status:** ✅ Built · ✅ Tested locally · 🔜 Ready for Streamlit deployment
