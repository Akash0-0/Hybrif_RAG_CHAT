import os
import dotenv
from pinecone import Pinecone
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

dotenv.load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-rag-index")


def retrieve_docs(query: str, user_id: str, k: int = 5) -> list[dict]:
    """
    Hybrid retrieval (vector + BM25) scoped to a single user's Pinecone namespace.

    Args:
        query:   The search query string.
        user_id: Pinecone namespace that isolates this user's documents.
        k:       Number of results to return.

    Returns:
        List of dicts with keys 'page_content' and 'metadata'.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)

    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )

    # ── 1. Vector retrieval via Pinecone ──────────────────────────────────────
    query_embedding = embedding_model.embed_query(query)
    vector_results = index.query(
        vector=query_embedding,
        top_k=k,
        namespace=user_id,
        include_metadata=True,
    )

    vector_docs: list[Document] = []
    for match in vector_results.get("matches", []):
        meta = match.get("metadata", {})
        text = meta.pop("text", "")  # text was stored in metadata at ingest time
        vector_docs.append(Document(page_content=text, metadata=meta))

    # ── 2. BM25 retrieval over the same namespace corpus ─────────────────────
    # Fetch all documents in this namespace for BM25 (up to 10 000 — adjust as needed)
    fetch_response = index.query(
        vector=[0.0] * 4096,       # dummy vector — we only want texts
        top_k=10_000,
        namespace=user_id,
        include_metadata=True,
    )

    all_texts: list[str] = []
    all_metas: list[dict] = []
    for match in fetch_response.get("matches", []):
        meta = dict(match.get("metadata", {}))
        text = meta.pop("text", "")
        all_texts.append(text)
        all_metas.append(meta)

    bm25_docs: list[Document] = []
    if all_texts:
        bm25_retriever = BM25Retriever.from_texts(texts=all_texts, metadatas=all_metas)
        bm25_retriever.k = max(3, k // 2)
        bm25_docs = bm25_retriever.invoke(query)

    # ── 3. Merge & de-duplicate by page_content ───────────────────────────────
    seen: set[str] = set()
    merged: list[dict] = []

    # Vector results carry higher weight — insert first
    for doc in vector_docs + bm25_docs:
        content = getattr(doc, "page_content", str(doc)).strip()
        if content and content not in seen:
            seen.add(content)
            merged.append(
                {
                    "page_content": content,
                    "metadata": getattr(doc, "metadata", {}),
                }
            )
        if len(merged) >= k:
            break

    return merged
