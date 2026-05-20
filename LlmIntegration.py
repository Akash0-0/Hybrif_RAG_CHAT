from langchain_nvidia_ai_endpoints import ChatNVIDIA
from Retrieval import retrieve_docs
from Ingestion import ingest_docs
import os
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

dotenv.load_dotenv()

chat_history = []


def rewrite_question(user_question, history):
    if not history:
        return user_question

    model = ChatNVIDIA(
        model="nvidia/nemotron-mini-4b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.2,
        top_p=0.7,
        max_completion_tokens=512,
    )

    messages = [
        SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
    ] + history + [HumanMessage(content=f"New question: {user_question}")]

    result = model.invoke(messages)
    return result.content.strip()


def generate_answer(user_question, docs):
    doc_text = "\n\n".join([f"Source: {doc['metadata'].get('source', 'unknown')}\n{doc['page_content']}" for doc in docs])
    prompt = f"Answer the user question using only the documents below. If the documents are not sufficient, say so clearly.\n\nUser question: {user_question}\n\nDocuments:\n{doc_text}"

    model = ChatNVIDIA(
        model="nvidia/nemotron-mini-4b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.2,
        top_p=0.7,
        max_completion_tokens=512,
    )

    response = model.invoke([
        SystemMessage(content="You are a helpful assistant that answers user questions from provided documents."),
        HumanMessage(content=prompt),
    ])
    return response.content.strip()


def history_aware_generation(user_question, chat_history):
    if chat_history:
        search_question = rewrite_question(user_question, chat_history)
    else:
        search_question = user_question

    docs = retrieve_docs(search_question)
    answer = generate_answer(user_question, docs)
    return docs, answer


if __name__ == '__main__':
    docs_to_ingest = input("Enter the document paths to ingest, separated by commas: ").split(",")
    docs_to_ingest = [path.strip() for path in docs_to_ingest if path.strip()]

    if not docs_to_ingest:
        print("No paths entered. Exiting.")
        exit(0)

    ingest_docs(docs_to_ingest)
    print(f"Ingested {len(docs_to_ingest)} document path(s).\n")

    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        docs, answer = history_aware_generation(user_question, chat_history)
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=answer))

        print("\n---\nAnswer:\n")
        print(answer)
        #print("\nRetrieved documents:\n")
        #for i, doc in enumerate(docs, 1):
        #    content = doc['page_content']
        #    preview = content[:400].replace('\n', ' ')
        #    suffix = '...' if len(content) > 400 else ''
        #    print(f"[{i}] {doc['metadata'].get('source', 'unknown')}: {preview}{suffix}\n")
        #print("---\n")




