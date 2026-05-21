---
title: Hybrid RAG Chatbot
emoji: ⚡
colorFrom: purple
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
sdk_version: 5.16.0
python_version: 3.11
---

# ⚡ Hybrid RAG Chatbot

A production-grade, multi-user RAG chatbot featuring:

- **Hybrid Retrieval** — cosine vector search (NVIDIA `nv-embed-v1`) + BM25, merged and de-duplicated
- **History-aware Q&A** — follow-up questions are rewritten into standalone queries before retrieval
- **Per-user isolation** — every browser session gets its own Pinecone namespace; no data leaks between users
- **Cloud vector DB** — [Pinecone](https://www.pinecone.io/) serverless (free tier supported)
- **NVIDIA LLMs** — `nemotron-mini-4b-instruct` for both rewriting and answer generation

---

## 🗂 File Structure

```
app.py            ← Gradio UI + Flask thread launcher (HF Spaces entrypoint)
api.py            ← Flask REST API  (/ingest, /chat, /reset, /health)
Ingestion.py      ← Document loading, chunking, embedding → Pinecone upsert
Retrieval.py      ← Hybrid vector + BM25 retrieval from Pinecone
LlmIntegration.py ← Question rewriting + answer generation
requirements.txt
```

---

## 🔑 Required Secrets (HF Spaces → Settings → Repository secrets)

| Secret name          | Where to get it                                          |
|----------------------|----------------------------------------------------------|
| `NVIDIA_API_KEY`     | [build.nvidia.com](https://build.nvidia.com) → API Keys |
| `PINECONE_API_KEY`   | [app.pinecone.io](https://app.pinecone.io) → API Keys   |
| `PINECONE_INDEX_NAME`| Name of your Pinecone index (e.g. `hybrid-rag-index`)   |

> **Pinecone index settings:** dimension = **4096**, metric = **cosine**, cloud = **AWS us-east-1**
> The app auto-creates the index if it doesn't exist.

---

## 🚀 Local Development

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Create .env
echo "NVIDIA_API_KEY=nvapi-..." >> .env
echo "PINECONE_API_KEY=pcsk_..." >> .env
echo "PINECONE_INDEX_NAME=hybrid-rag-index" >> .env

# 3. Run
python app.py
# Gradio → http://localhost:7860
# Flask  → http://localhost:7861
```

---

## 🏗 Architecture

```
Browser
  │
  ▼
Gradio UI (port 7860)
  │  HTTP
  ▼
Flask API (port 7861)
  ├── /ingest ──► Ingestion.py ──► Pinecone (namespace = user_id)
  ├── /chat   ──► LlmIntegration.py
  │                  ├── rewrite_question()  → ChatNVIDIA
  │                  ├── retrieve_docs()     → Pinecone vector + BM25
  │                  └── generate_answer()   → ChatNVIDIA
  └── /reset  ──► clears in-memory chat history
```

### User isolation
Each session is assigned a UUID (`user_id`) on first ingestion. This becomes the **Pinecone namespace** for all that user's vectors. Queries are always scoped to the namespace — different users never see each other's documents.
