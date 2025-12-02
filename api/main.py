from fastapi import FastAPI
from pydantic import BaseModel

from rag.bootstrap import initialize_kb
from rag.pipeline import answer_question


app = FastAPI(title="Appliance Troubleshooter RAG API")


class QuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
def on_startup():
    """
    Initialise la base de connaissances au lancement de l'API.
    """
    initialize_kb()


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    """
    Endpoint principal : prend une question en entrée et renvoie une réponse RAG.
    """
    result = answer_question(payload.question)
    return result