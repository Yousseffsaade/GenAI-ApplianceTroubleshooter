from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Modèle léger, local et gratuit
_LLM_NAME = "distilgpt2"

print(f"[llm] Chargement du modèle {_LLM_NAME}...")
_tokenizer = AutoTokenizer.from_pretrained(_LLM_NAME)
_model = AutoModelForCausalLM.from_pretrained(_LLM_NAME)
_model.eval()
print("[llm] Modèle chargé.")


def generate_answer(prompt: str, max_new_tokens: int = 150) -> str:
    """
    Génère une réponse texte à partir d'un prompt.
    Le prompt inclut déjà le contexte et la question (RAG pipeline).
    """
    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=_tokenizer.eos_token_id,
        )

    full_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # on coupe tout ce qui précède "Answer:" si présent
    if "Answer:" in full_text:
        answer = full_text.split("Answer:", maxsplit=1)[-1].strip()
    else:
        answer = full_text.strip()

    return answer