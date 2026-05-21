import os
import dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hybrid-rag-index")
EMBEDDING_DIM = 4096  # nvidia/nv-embed-v1 output dimension


def get_pinecone_index():
    """Initialize Pinecone client and return the index (creates if missing)."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def ingest_docs(doc_paths: list[str], user_id: str) -> int:
    """
    Ingest documents into Pinecone under a per-user namespace.

    Args:
        doc_paths: List of file paths (PDF or text).
        user_id:   Unique session/user identifier used as Pinecone namespace.

    Returns:
        Number of chunks ingested.
    """
    index = get_pinecone_index()

    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    ids, metadatas, documents = [], [], []
    chunk_counter = 0

    for doc_path in doc_paths:
        doc_path = doc_path.strip()
        if not doc_path:
            continue

        loader = PyPDFLoader(doc_path) if doc_path.lower().endswith(".pdf") else TextLoader(doc_path)
        loaded_docs = loader.load()
        if not isinstance(loaded_docs, list):
            loaded_docs = [loaded_docs]

        for source_doc in loaded_docs:
            text = getattr(source_doc, "page_content", str(source_doc))
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                ids.append(f"{os.path.basename(doc_path)}_{chunk_counter}")
                chunk_counter += 1
                metadatas.append({"source": os.path.basename(doc_path), "user_id": user_id})
                documents.append(chunk)

    if not documents:
        return 0

    # Embed in batches of 96 (NVIDIA endpoint limit)
    BATCH = 96
    vectors = []
    for i in range(0, len(documents), BATCH):
        batch_docs = documents[i : i + BATCH]
        batch_embeddings = embedding_model.embed_documents(batch_docs)
        for j, (doc_id, embedding, meta, doc_text) in enumerate(
            zip(ids[i : i + BATCH], batch_embeddings, metadatas[i : i + BATCH], batch_docs)
        ):
            vectors.append(
                {
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {**meta, "text": doc_text},
                }
            )

    # Upsert in batches of 100 (Pinecone limit)
    UPSERT_BATCH = 100
    for i in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[i : i + UPSERT_BATCH], namespace=user_id)

    return len(documents)
