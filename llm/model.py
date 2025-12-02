from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Modèle léger, local et gratuit
_LLM_NAME = "distilgpt2"

print(f"[llm] Chargement du modèle {_LLM_NAME}...")
_tokenizer = AutoTokenizer.from_pretrained(_LLM_NAME)
_model = AutoModelForCausalLM.from_pretrained(_LLM_NAME)
_model.eval()
print("[llm] Modèle chargé.")


def generate_answer(context_and_question: str, max_new_tokens: int = 120) -> str:
    """
    Génère une réponse claire à partir du contexte + question du pipeline RAG
    """

    prompt = f"""
You are an appliance troubleshooting expert.

Use ONLY the context below to answer the question clearly and practically.

Context:
{context_and_question}

Answer:
"""

    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.eos_token_id,
        )

    full_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # On récupère strictement ce qui vient après "Answer:"
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", 1)[1]
    else:
        answer = full_text

    # Nettoyage du résultat
    answer = answer.strip()

    # Empêche les répétitions ou textes interminables
    answer = answer.split("\n")[0].strip()

    if len(answer) < 3:
        answer = "I could not generate a useful answer from the current context."

    return answer