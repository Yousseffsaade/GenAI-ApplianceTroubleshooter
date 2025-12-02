import chromadb

# Client en mémoire (tout est perdu quand le programme s'arrête, mais c'est parfait pour dev)
_client = chromadb.EphemeralClient()

# Le nom doit faire au moins 3 caractères
_collection = _client.get_or_create_collection(name="appliance_troubleshooting_kb")


def add_documents(texts, ids, embeddings):
    """
    Ajoute des documents + embeddings dans la collection Chroma.
    """
    _collection.add(
        documents=texts,
        ids=ids,
        embeddings=embeddings,
    )


def query_by_embeddings(query_embeddings, k: int = 3):
    """
    Fait une requête sur la base Chroma à partir d'embeddings de requête.
    """
    return _collection.query(
        query_embeddings=query_embeddings,
        n_results=k,
    )


def count_documents() -> int:
    """
    Retourne le nombre de documents dans la collection.
    """
    return _collection.count()