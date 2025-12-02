from sentence_transformers import SentenceTransformer

# On charge UN SEUL modèle globalement
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts):
    """
    Prend une liste de textes et retourne une liste de vecteurs (embeddings).
    """
    if isinstance(texts, str):
        texts = [texts]
    return _model.encode(texts).tolist()