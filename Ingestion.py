import os
import dotenv
import chromadb
from chromadb.config import Settings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


def ingest_docs(docs, persist_directory="chroma_db", collection_name="Hybrid_RAG_Index"):
    """Ingest documents into a local Chroma collection using NVIDIA embeddings."""
    client = chromadb.Client(settings=Settings(is_persistent=True, persist_directory=persist_directory))
    collection = client.get_or_create_collection(name=collection_name)

    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    ids = []
    metadatas = []
    documents = []
    chunk_counter = 0

    for doc_path in docs:
        doc_path = doc_path.strip()
        if not doc_path:
            continue

        if doc_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(doc_path)
        else:
            loader = TextLoader(doc_path)

        loaded_docs = loader.load()
        if not isinstance(loaded_docs, list):
            loaded_docs = [loaded_docs]

        for source_doc in loaded_docs:
            text = getattr(source_doc, 'page_content', str(source_doc))
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                ids.append(f"{os.path.basename(doc_path)}_{chunk_counter}")
                chunk_counter += 1
                metadatas.append({
                    "source": os.path.basename(doc_path),
                })
                documents.append(chunk)

    if not documents:
        return None

    embeddings = embedding_model.embed_documents(documents)
    collection.add(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
    # New Chroma client persists automatically when using is_persistent=True
    return collection



