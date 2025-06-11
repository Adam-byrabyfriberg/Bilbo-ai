import requests
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

# Ladda sentence-transformer för semantisk sökning
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extern LLM-modell via Replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_MODEL = "mistralai/mistral-7b-instruct-v0.1"

def ask_question(index, question):
    # Skapa embedding för frågan
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Ranka dokument
    hits = util.semantic_search(question_embedding, index["embeddings"], top_k=3)[0]
    context = "\n\n".join([index["texts"][hit["corpus_id"]] for hit in hits])

    # Bygg prompt
    prompt = f"""
Du är en AI-assistent som endast får svara baserat på informationen nedan.

{context}

Fråga: {question}
Svar:
"""

    # Skicka prompt till Replicate
    response = requests.post(
        "https://api.replicate.com/v1/completions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "model": REPLICATE_MODEL,
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7
        }
    )

    if response.ok:
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    else:
        return "Jag kunde tyvärr inte få fram ett svar just nu."
