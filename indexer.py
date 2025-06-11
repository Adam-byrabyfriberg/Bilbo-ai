# --- Fil: indexer.py ---
import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import torch

INDEX_FILE = "index.pt"
model = SentenceTransformer("all-MiniLM-L6-v2")

# URL att indexera
TARGET_URL = "https://byrabyfriberg.se"


def fetch_text_from_site():
    try:
        response = requests.get(TARGET_URL)
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
        return lines[:100]  # Begränsa till max 100 rader
    except Exception as e:
        print("Kunde inte hämta text:", e)
        return []


def build_or_load_index():
    if os.path.exists(INDEX_FILE):
        return torch.load(INDEX_FILE)

    print("Bygger nytt index från", TARGET_URL)
    texts = fetch_text_from_site()
    embeddings = model.encode(texts, convert_to_tensor=True)
    index = {"texts": texts, "embeddings": embeddings}
    torch.save(index, INDEX_FILE)
    return index
