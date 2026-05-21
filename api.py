"""
Flask API backend for Hybrid RAG chatbot.
Exposes two endpoints consumed by the Gradio frontend:
  POST /ingest  — upload & index documents for a session user
  POST /chat    — ask a question with optional history
"""

import os
import uuid
import tempfile
import dotenv
from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage, AIMessage

from Ingestion import ingest_docs
from LlmIntegration import history_aware_generation

dotenv.load_dotenv()

app = Flask(__name__)

# In-memory chat history store: { user_id: [LangChain message, ...] }
# For production consider Redis; fine for HF Spaces (single-process)
_chat_histories: dict[str, list] = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Expects multipart/form-data:
      - files: one or more files (PDF / .txt)
      - user_id: (optional) existing session ID; a new one is created if absent

    Returns JSON:
      { "user_id": "...", "chunks_ingested": N, "message": "..." }
    """
    user_id = request.form.get("user_id") or str(uuid.uuid4())
    uploaded_files = request.files.getlist("files")

    if not uploaded_files:
        return jsonify({"error": "No files provided"}), 400

    saved_paths: list[str] = []
    tmp_dir = tempfile.mkdtemp()

    for f in uploaded_files:
        safe_name = os.path.basename(f.filename or "upload")
        dest = os.path.join(tmp_dir, safe_name)
        f.save(dest)
        saved_paths.append(dest)

    try:
        count = ingest_docs(saved_paths, user_id=user_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    # Reset chat history when new docs are ingested
    _chat_histories[user_id] = []

    return jsonify(
        {
            "user_id": user_id,
            "chunks_ingested": count,
            "message": f"Successfully ingested {count} chunks from {len(saved_paths)} file(s).",
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    """
    Expects JSON body:
      { "user_id": "...", "question": "..." }

    Returns JSON:
      {
        "answer": "...",
        "sources": ["file1.pdf", ...],
        "user_id": "..."
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "").strip()
    question = data.get("question", "").strip()

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400

    history = _chat_histories.get(user_id, [])

    try:
        docs, answer = history_aware_generation(question, history, user_id=user_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Update history
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    _chat_histories[user_id] = history

    sources = list({doc["metadata"].get("source", "unknown") for doc in docs})

    return jsonify(
        {
            "answer": answer,
            "sources": sources,
            "user_id": user_id,
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """
    Clears chat history for a user (does NOT delete vectors from Pinecone).
    Expects JSON: { "user_id": "..." }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "").strip()
    if user_id in _chat_histories:
        _chat_histories[user_id] = []
    return jsonify({"message": "Chat history cleared.", "user_id": user_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7861, debug=False)
