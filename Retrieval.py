import os
import dotenv
import chromadb
from chromadb.config import Settings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
dotenv.load_dotenv()


def retrieve_docs(query, persist_directory="chroma_db", collection_name="Hybrid_RAG_Index", k=5):
    """Retrieve relevant document chunks from the Chroma collection."""
    client = chromadb.Client(settings=Settings(is_persistent=True, persist_directory=persist_directory))
    collection = client.get_or_create_collection(name=collection_name)
    #vectors first
    embedding_model = NVIDIAEmbeddings(
        model="nvidia/nv-embed-v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )

    collection_data = collection.get(include=["documents", "metadatas"])

    vectorstore = Chroma.from_texts(
        texts=collection_data["documents"],
        embedding=embedding_model,
        metadatas=collection_data["metadatas"],
        collection_metadata={"hnsw:space": "cosine"}
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    
    #not bm25 

    bm25_retriever = BM25Retriever.from_texts(
        texts=collection_data["documents"],
        metadatas=collection_data["metadatas"]
    )
    bm25_retriever.k = 3


    # Hybrid retrieval

    HybridRetriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.7, 0.6])
    results = HybridRetriever.invoke(query)

    docs = []
    for document in results:
        if isinstance(document, str):
            docs.append({
                "page_content": document,
                "metadata": {},
            })
        else:
            docs.append({
                "page_content": getattr(document, "page_content", str(document)),
                "metadata": getattr(document, "metadata", {}),
            })

    return docs
