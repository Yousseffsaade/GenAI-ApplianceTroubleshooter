from .loader import load_documents
from .embeddings import embed
from .vectorstore import add_documents, count_documents


def initialize_kb():
    """
    Charge les documents, calcule leurs embeddings, et les ajoute à Chroma.
    Appelée une seule fois au démarrage de l'API.
    """
    if count_documents() > 0:
        # Déjà initialisé dans ce processus
        print("[bootstrap] KB déjà initialisée, skip.")
        return

    docs = load_documents()

    # fallback si aucun fichier trouvé
    if not docs:
        print("[bootstrap] Aucun fichier trouvé dans data/raw, utilisation de docs par défaut.")
        docs = [
            {"id": "doc1", "text": "The washing machine is leaking water from the bottom."},
            {"id": "doc2", "text": "The refrigerator is not cooling properly."},
            {"id": "doc3", "text": "The oven does not heat up when turned on."},
        ]

    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    print(f"[bootstrap] Initialisation de la KB avec {len(texts)} documents...")
    embeddings = embed(texts)
    add_documents(texts, ids, embeddings)
    print("[bootstrap] KB initialisée.")