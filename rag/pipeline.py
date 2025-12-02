from .retriever import retrieve_similar_docs
from llm.model import generate_answer


def answer_question(question: str, top_k: int = 3):
    """
    Pipeline RAG :
    - R : on récupère les documents les plus proches dans Chroma
    - A : on fabrique un prompt avec le contexte
    - G : on demande au LLM de générer une réponse basée sur ce contexte
    """
    results = retrieve_similar_docs(question, top_k=top_k)
    docs = results.get("documents", [[]])[0]  # liste de textes
    if not docs:
        return {
            "answer": "Je ne trouve aucune information pertinente dans la base de connaissances.",
            "sources": [],
        }

    # Construit le contexte sous forme de puces
    context = "\n".join(f"- {d}" for d in docs)

    prompt = f"""You are an appliance troubleshooting assistant.
Use ONLY the following context to answer the user's question.
If the answer is not clearly in the context, say you are not sure.

Context:
{context}

Question: {question}
Answer:"""

    answer_text = generate_answer(prompt)

    return {
        "answer": answer_text,
        "sources": docs,
    }