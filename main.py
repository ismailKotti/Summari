import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Liste des modèles à tester dans l'ordre (on exclut gemini-2.5-flash !)
CANDIDATE_MODELS = [
    "gemini-3.0-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

class TextInput(BaseModel):
    text: str
    min_length: int = 20
    max_length: int = 60

@app.post("/summarize")
def summarize_text(data: TextInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="La variable d'environnement GEMINI_API_KEY n'est pas configurée.",
        )

    prompt = (
        f"Summarize the following text clearly and concisely in approximately "
        f"{data.min_length} to {data.max_length} words. "
        f"Return ONLY the summary text without any introductory phrase:\n\n{data.text}"
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    last_error = ""
    response = None

    # On teste automatiquement les modèles jusqu'à ce que l'un réponde 200 OK
    for model_name in CANDIDATE_MODELS:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(api_url, json=payload, timeout=30)
            if resp.status_code == 200:
                response = resp
                break
            else:
                last_error = f"[{model_name} -> {resp.status_code}]: {resp.text}"
        except Exception as e:
            last_error = str(e)

    if response is None:
        raise HTTPException(status_code=500, detail=f"Aucun modèle n'a répondu. Dernière erreur : {last_error}")

    result = response.json()
    try:
        summary_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail=f"Réponse inattendue : {result}")

    original_words = len(data.text.split())
    summary_words = len(summary_text.split())

    return {
        "summary": summary_text,
        "original_word_count": original_words,
        "summary_word_count": summary_words
    }