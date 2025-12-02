from .embeddings import embed
from .vectorstore import query_by_embeddings


def retrieve_similar_docs(question: str, top_k: int = 3):
    """
    Encode la question en embedding, interroge Chroma,
    et retourne le résultat brut de Chroma.
    """
    query_emb = embed([question])  # liste de 1 vecteur
    results = query_by_embeddings(query_emb, k=top_k)
    return results