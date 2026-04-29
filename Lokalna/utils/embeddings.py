import logging
import requests
from config import EMBEDDING_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

def ucitaj_model():
    """Ollama sama upravlja modelima, pa ovdje samo vraćamo naziv modela."""
    logger.info(f"Korištenje Ollama embedding modela: {EMBEDDING_MODEL}")
    return EMBEDDING_MODEL

def generiraj_embeddings(model_name: str, tekstovi: list[str]) -> list[list[float]]:
    """Šalje zahtjev Ollama API-ju za generiranje embeddinga."""
    if not tekstovi:
        return []

    vektori = []
    try:
        for tekst in tekstovi:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={
                    "model": model_name,
                    "prompt": tekst
                },
                timeout=60
            )
            response.raise_for_status()
            vektori.append(response.json()["embedding"])
        
        return vektori
    except Exception as e:
        logger.error(f"Greška pri generiranju embeddinga preko Ollame: {e}")
        # Ako Ollama vrati grešku, aplikacija će prikazati tvoju poruku o grešci
        raise

def generiraj_jedan_embedding(model_name: str, tekst: str) -> list[float]:
    """Generira embedding za jedan tekst (upit korisnika)."""
    res = generiraj_embeddings(model_name, [tekst])
    return res[0] if res else []