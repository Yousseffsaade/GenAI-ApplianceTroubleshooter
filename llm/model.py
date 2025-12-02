"""
Module LLM optimisé pour le système RAG de dépannage d'appareils
Compatible avec la structure existante du projet
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import warnings
warnings.filterwarnings('ignore')

# Configuration globale
MODEL_TYPE = "flan-t5"  # Changez pour "distilgpt2" ou "fallback" si nécessaire
_model = None
_tokenizer = None
_initialized = False

def _initialize_model():
    """Initialisation paresseuse du modèle"""
    global _model, _tokenizer, _initialized, MODEL_TYPE
    
    if _initialized:
        return
    
    print(f"[llm] Chargement du modèle {MODEL_TYPE}...")
    
    try:
        if MODEL_TYPE == "flan-t5":
            # Flan-T5 est EXCELLENT pour Q&A et instructions
            model_name = "google/flan-t5-small"  # 80M params, ~300MB
            # Alternative plus puissante : "google/flan-t5-base" (250M params, ~1GB)
            
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            _model.eval()
            print(f"[llm] Modèle Flan-T5 chargé avec succès")
            
        elif MODEL_TYPE == "distilgpt2":
            # DistilGPT-2 : plus léger mais moins performant
            model_name = "distilgpt2"
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Fix du padding token
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            _model.eval()
            print(f"[llm] Modèle DistilGPT-2 chargé")
            
        else:
            # Mode fallback sans ML
            print("[llm] Mode fallback activé (extraction basée sur règles)")
            
    except Exception as e:
        print(f"[llm] Erreur lors du chargement du modèle: {e}")
        print("[llm] Basculement en mode fallback")
        MODEL_TYPE = "fallback"
        _model = None
        _tokenizer = None
    
    _initialized = True


def generate_answer(context_and_question: str, max_new_tokens: int = 120) -> str:
    """
    Génère une réponse claire à partir du contexte + question du pipeline RAG
    Compatible avec l'interface existante
    """
    _initialize_model()
    
    if MODEL_TYPE == "flan-t5" and _model is not None:
        return _generate_with_flan_t5(context_and_question, max_new_tokens)
    elif MODEL_TYPE == "distilgpt2" and _model is not None:
        return _generate_with_distilgpt2(context_and_question, max_new_tokens)
    else:
        return _generate_with_fallback(context_and_question)


def _generate_with_flan_t5(prompt: str, max_new_tokens: int) -> str:
    """
    Génération avec Flan-T5 (recommandé)
    Ce modèle comprend très bien les instructions
    """
    # Reformater le prompt pour Flan-T5
    if "Context:" in prompt and "Question:" in prompt:
        # Extraire les parties
        parts = prompt.split("Question:")
        if len(parts) >= 2:
            context_part = parts[0].replace("Context:", "").strip()
            question_part = parts[1].replace("Answer:", "").strip()
            
            # Prompt optimisé pour Flan-T5
            clean_prompt = f"Answer this question based on the context. Context: {context_part} Question: {question_part}"
        else:
            clean_prompt = prompt
    else:
        clean_prompt = prompt
    
    # Tokenisation
    inputs = _tokenizer(
        clean_prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Génération
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            num_beams=2  # Un peu de beam search améliore la qualité
        )
    
    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Validation de la réponse
    if len(answer.strip()) < 5:
        return _generate_with_fallback(prompt)
    
    return answer.strip()


def _generate_with_distilgpt2(prompt: str, max_new_tokens: int) -> str:
    """
    Génération avec DistilGPT-2
    Moins performant mais plus léger
    """
    # Simplifier le prompt pour DistilGPT-2
    if "Context:" in prompt:
        # Extraire juste le contexte principal
        context_section = prompt.split("Context:")[1].split("Question:")[0].strip()
        question_section = prompt.split("Question:")[1].replace("Answer:", "").strip() if "Question:" in prompt else ""
        
        # Prompt simple et direct
        simple_prompt = f"The {question_section} because {context_section[:200]}"
    else:
        simple_prompt = prompt[:300]  # Limiter la taille
    
    inputs = _tokenizer(
        simple_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=_tokenizer.pad_token_id
        )
    
    full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraire uniquement la partie générée
    answer = full_text[len(simple_prompt):].strip()
    
    # Nettoyer
    answer = answer.split("\n")[0].strip()
    
    if len(answer) < 10:
        return _generate_with_fallback(prompt)
    
    return answer


def _generate_with_fallback(prompt: str) -> str:
    """
    Mode fallback intelligent sans modèle ML
    Extrait et reformule les informations du contexte
    """
    prompt_lower = prompt.lower()
    
    # Dictionnaire des problèmes et solutions courantes
    appliance_solutions = {
        # Lave-linge
        ("washing machine", "leak"): "Check the door seal for damage, inspect the drain hose connection, and clean the filter. These are the most common causes of water leaks.",
        ("washing machine", "water under"): "Water under the machine typically comes from a damaged door seal, loose drain hose, or clogged filter. Inspect each component.",
        ("washing machine", "not drain"): "Check if the drain hose is clogged or kinked. Clean the pump filter and ensure the drain pipe isn't blocked.",
        
        # Réfrigérateur
        ("refrigerator", "not cool"): "Clean the condenser coils, check the thermostat settings, and ensure air vents aren't blocked. Verify the door seals properly.",
        ("fridge", "not cool"): "Dirty condenser coils, faulty thermostat, or blocked air vents are common causes. Check and clean these components.",
        ("refrigerator", "noise"): "Check if the appliance is level. Clean the condenser coils and ensure nothing is touching the back wall.",
        
        # Four
        ("oven", "not heat"): "The heating element may be broken, the temperature sensor could be faulty, or there might be a power supply issue. Test each component.",
        ("oven", "not work"): "Check the heating element, temperature sensor, and power connection. Ensure the door closes properly.",
        
        # Lave-vaisselle
        ("dishwasher", "dirty dish"): "Clean the spray arms, check water pressure, and use the correct amount of detergent. Pre-rinse heavily soiled items.",
        ("dishwasher", "not clean"): "Clogged spray arms, low water pressure, or insufficient detergent are common causes. Check and clean the filter too.",
        
        # Sèche-linge
        ("dryer", "wet cloth"): "Clean the lint filter, check the exhaust vent for blockages, and verify the heating element is working. Don't overload the machine.",
        ("dryer", "not dry"): "A blocked vent, broken heating element, or overloading are typical causes. Clean all vents and filters thoroughly.",
    }
    
    # Recherche de la meilleure correspondance
    for (appliance, problem), solution in appliance_solutions.items():
        if appliance in prompt_lower and problem in prompt_lower:
            return solution
    
    # Extraction intelligente du contexte
    if "Context:" in prompt:
        context = prompt.split("Context:")[1].split("Question:")[0] if "Question:" in prompt else prompt.split("Context:")[1]
        
        # Chercher les phrases clés dans le contexte
        sentences = context.split(".")
        for sentence in sentences:
            sent_lower = sentence.lower().strip()
            
            # Phrases indiquant des causes
            if any(marker in sent_lower for marker in ["possible causes", "can be caused", "may be due", "common issues"]):
                # Extraire et reformuler cette information
                causes = sentence.strip()
                if "include" in causes.lower():
                    parts = causes.split("include", 1)[1].strip()
                    return f"The issue is likely due to: {parts}. Check each of these components."
                return f"Based on the information: {causes}"
            
            # Phrases avec des actions recommandées
            if any(action in sent_lower for action in ["check", "inspect", "clean", "replace", "verify"]):
                return sentence.strip()
    
    # Réponse générique basée sur les mots-clés détectés
    if "leak" in prompt_lower or "water" in prompt_lower:
        return "Check for damaged seals, loose connections, and clogged filters. These are common causes of water leaks."
    elif "not cool" in prompt_lower or "not heat" in prompt_lower:
        return "Temperature issues are often caused by faulty thermostats, broken heating/cooling elements, or blocked vents."
    elif "noise" in prompt_lower or "loud" in prompt_lower:
        return "Unusual noises can indicate unbalanced loads, worn bearings, or loose components. Check if the appliance is level."
    elif "not start" in prompt_lower or "not work" in prompt_lower:
        return "Verify the power supply, check if the door is properly closed, and inspect the control panel for error codes."
    
    # Réponse par défaut
    return "Based on the symptoms described, inspect the main components mentioned in your manual. If the issue persists, consult a qualified technician."


# Test du module si exécuté directement
if __name__ == "__main__":
    print("\n=== Test du module LLM ===\n")
    
    # Test avec un contexte typique de votre système
    test_prompt = """You are an appliance troubleshooting assistant.
Use ONLY the following context to answer the user's question.
If the answer is not clearly in the context, say you are not sure.

Context:
- The washing machine is leaking water from the bottom. Possible causes include a damaged door seal, a loose drain hose, or a clogged filter.
- The refrigerator is not cooling properly. Common issues are dirty condenser coils, a faulty thermostat, or blocked air vents inside the fridge.

Question: My washing machine has water under it
Answer:"""
    
    print("Prompt de test:")
    print(test_prompt[:200] + "...")
    print("\nGénération de la réponse...")
    
    answer = generate_answer(test_prompt)
    
    print(f"\nRéponse générée:\n{answer}")
    print("\n=== Test terminé ===")