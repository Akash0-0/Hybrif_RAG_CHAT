import os
import dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Retrieval import retrieve_docs

dotenv.load_dotenv()


def rewrite_question(user_question: str, history: list) -> str:
    """Rewrite the user's question to be standalone, given prior chat history."""
    if not history:
        return user_question

    model = ChatNVIDIA(
        model="nvidia/nemotron-mini-4b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.2,
        top_p=0.7,
        max_tokens=512,
    )

    messages = [
        SystemMessage(
            content=(
                "Given the chat history, rewrite the new question to be standalone "
                "and searchable. Return only the rewritten question, nothing else."
            )
        ),
    ] + history + [HumanMessage(content=f"New question: {user_question}")]

    result = model.invoke(messages)
    return result.content.strip()


def generate_answer(user_question: str, docs: list[dict]) -> str:
    """Generate an answer grounded in the retrieved documents."""
    if not docs:
        return "I couldn't find any relevant documents to answer your question. Please make sure you've uploaded relevant files first."

    doc_text = "\n\n".join(
        [
            f"Source: {doc['metadata'].get('source', 'unknown')}\n{doc['page_content']}"
            for doc in docs
        ]
    )
    prompt = (
        "Answer the user question using only the documents below. "
        "If the documents are not sufficient, say so clearly.\n\n"
        f"User question: {user_question}\n\nDocuments:\n{doc_text}"
    )

    model = ChatNVIDIA(
        model="nvidia/nemotron-mini-4b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.2,
        top_p=0.7,
        max_tokens=512,
    )

    response = model.invoke(
        [
            SystemMessage(content="You are a helpful assistant that answers user questions from provided documents."),
            HumanMessage(content=prompt),
        ]
    )
    return response.content.strip()


def history_aware_generation(
    user_question: str,
    chat_history: list,
    user_id: str,
) -> tuple[list[dict], str]:
    """
    Full RAG pipeline: optional question rewriting → retrieval → answer generation.

    Args:
        user_question: The raw question from the user.
        chat_history:  List of LangChain HumanMessage / AIMessage objects.
        user_id:       Pinecone namespace for this user's documents.

    Returns:
        (retrieved_docs, answer_string)
    """
    search_question = (
        rewrite_question(user_question, chat_history) if chat_history else user_question
    )
    docs = retrieve_docs(search_question, user_id=user_id)
    answer = generate_answer(user_question, docs)
    return docs, answer
