"""
Gradio frontend for Hybrid RAG Chatbot.
Runs the Flask API in a background thread so HF Spaces only needs one process.
"""

import threading
import requests
import gradio as gr
from api import app as flask_app  # import Flask app

API_BASE = "http://127.0.0.1:7861"


# ── Start Flask in a daemon thread ────────────────────────────────────────────
def _run_flask():
    flask_app.run(host="0.0.0.0", port=7861, debug=False, use_reloader=False)


flask_thread = threading.Thread(target=_run_flask, daemon=True)
flask_thread.start()


# ── Helper: call Flask API ────────────────────────────────────────────────────
def _post(endpoint: str, **kwargs):
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


# ── Gradio callback functions ─────────────────────────────────────────────────
def ingest_files(files, user_id: str):
    """Upload files to the /ingest endpoint and return status + (possibly new) user_id."""
    if not files:
        return user_id, "⚠️ Please upload at least one file."

    form_data = {"user_id": user_id} if user_id else {}
    open_files = [("files", (f.name.split("/")[-1], open(f.name, "rb"))) for f in files]

    try:
        resp = requests.post(
            f"{API_BASE}/ingest",
            data=form_data,
            files=open_files,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return user_id, f"❌ Error: {e}"
    finally:
        for _, (_, fh) in open_files:
            fh.close()

    new_uid = data.get("user_id", user_id)
    msg = data.get("message", "Done.")
    return new_uid, f"✅ {msg}\n\n🔑 Your session ID: `{new_uid}`"


def chat(user_message: str, history: list, user_id: str):
    """Send a question to /chat and stream the reply into the Gradio chatbot."""
    if not user_id:
        history = history or []
        history.append((user_message, "⚠️ Please ingest documents first to get a session ID."))
        return history, ""

    payload = {"user_id": user_id, "question": user_message}
    data, err = _post("/chat", json=payload)

    if err or not data:
        reply = f"❌ API error: {err}"
    else:
        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])
        source_line = (
            f"\n\n📎 **Sources:** {', '.join(sources)}" if sources else ""
        )
        reply = answer + source_line

    history = history or []
    history.append((user_message, reply))
    return history, ""


def reset_history(user_id: str):
    if not user_id:
        return "⚠️ No active session to reset."
    data, err = _post("/reset", json={"user_id": user_id})
    if err:
        return f"❌ {err}"
    return "🔄 Chat history cleared. Documents are still indexed."


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #1a1a26;
    --border: #2a2a3d;
    --accent: #7c6af7;
    --accent2: #4af0c4;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --radius: 12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.header-block {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(180deg, #0d0d1a 0%, transparent 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.header-block h1 {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}

.header-block p {
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    margin: 0;
}

/* Panels */
.gr-panel, .gr-box, .gr-group {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), #5b4fd4) !important;
    border: none !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.85 !important; }

.gr-button-secondary {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
}

/* Inputs */
input[type="text"], textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
}

/* Chatbot */
.gr-chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.gr-chatbot .message.user {
    background: linear-gradient(135deg, #2a2450, #1e1e35) !important;
    border-radius: 12px 12px 2px 12px !important;
    color: var(--text) !important;
}
.gr-chatbot .message.bot {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px 12px 12px 2px !important;
    color: var(--text) !important;
}

/* Labels */
label, .gr-label {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Status box */
.status-box textarea {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--accent2) !important;
}

/* Session ID box */
.session-box textarea {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.02em !important;
}

/* Dividers */
hr { border-color: var(--border) !important; }

/* Upload area */
.gr-file-upload {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface2) !important;
    transition: border-color 0.2s !important;
}
.gr-file-upload:hover { border-color: var(--accent) !important; }

/* Accordion */
.gr-accordion {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
"""

with gr.Blocks(css=CSS, title="Hybrid RAG Chatbot") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-block">
        <h1>⚡ Hybrid RAG Chatbot</h1>
        <p>Vector + BM25 retrieval · Per-session document isolation · History-aware Q&amp;A</p>
    </div>
    """)

    # ── Hidden session state ──────────────────────────────────────────────────
    user_id_state = gr.State("")

    # ── Layout ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left column — Document ingestion
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📂 Documents")

            file_upload = gr.File(
                label="Upload files (PDF / TXT)",
                file_count="multiple",
                file_types=[".pdf", ".txt"],
                height=160,
            )
            ingest_btn = gr.Button("🚀 Index Documents", variant="primary", size="lg")

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                elem_classes=["status-box"],
            )

            session_box = gr.Textbox(
                label="Session ID (auto-assigned)",
                interactive=False,
                lines=1,
                elem_classes=["session-box"],
                placeholder="Will appear after ingestion…",
            )

            with gr.Accordion("ℹ️ How it works", open=False):
                gr.Markdown(
                    """
**1. Upload** your PDF or text files and click *Index Documents*.  
**2. A unique session ID** is created — your vectors are stored in their own  
&nbsp;&nbsp;&nbsp;&nbsp;Pinecone namespace so they never mix with other users.  
**3. Ask questions** — the pipeline rewrites your query using chat history,  
&nbsp;&nbsp;&nbsp;&nbsp;retrieves chunks via hybrid search (cosine vector + BM25), and  
&nbsp;&nbsp;&nbsp;&nbsp;generates a grounded answer.  
**4. Reset** clears the conversation but keeps your indexed documents.
                    """
                )

        # Right column — Chat
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")

            chatbot = gr.Chatbot(
                label="",
                height=480,
                show_copy_button=True,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=rag"),
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    label="",
                    placeholder="Ask a question about your documents…",
                    lines=2,
                    scale=5,
                    show_label=False,
                    container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)

            reset_btn = gr.Button("🔄 Clear conversation history", variant="secondary", size="sm")
            reset_status = gr.Textbox(label="", interactive=False, lines=1, visible=True, container=False)

    # ── Event wiring ──────────────────────────────────────────────────────────

    # Ingest
    ingest_btn.click(
        fn=ingest_files,
        inputs=[file_upload, user_id_state],
        outputs=[user_id_state, status_box],
    ).then(
        fn=lambda uid: uid,
        inputs=[user_id_state],
        outputs=[session_box],
    )

    # Chat — button
    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, user_id_state],
        outputs=[chatbot, msg_box],
    )

    # Chat — Enter key
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, user_id_state],
        outputs=[chatbot, msg_box],
    )

    # Reset
    reset_btn.click(
        fn=reset_history,
        inputs=[user_id_state],
        outputs=[reset_status],
    ).then(
        fn=lambda: [],
        inputs=[],
        outputs=[chatbot],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
