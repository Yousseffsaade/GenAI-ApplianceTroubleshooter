import pathlib


def load_documents(raw_dir: str = "data/raw"):
    """
    Charge tous les fichiers .txt dans data/raw
    et les retourne sous forme de liste de dicts: {"id": ..., "text": ...}
    """
    base_path = pathlib.Path(raw_dir)
    docs = []

    if not base_path.exists():
        return docs

    for file_path in base_path.glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            doc_id = file_path.stem  # nom du fichier sans extension
            docs.append({"id": doc_id, "text": text})
        except Exception as e:
            print(f"[loader] Erreur en lisant {file_path}: {e}")

    return docs