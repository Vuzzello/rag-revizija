# Verzija: 3.0 | Ažurirano: 2026-04-29
# Lokalni embeddings putem sentence-transformers (BAAI/bge-m3)

import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

def generiraj_embeddings(model: SentenceTransformer, tekstovi: list[str]) -> list[list[float]]:
    """
    Generira embeddings lokalno koristeći učitani model.
    """
    if not tekstovi:
        logger.warning("Proslijeđena prazna lista tekstova.")
        return []

    try:
        # bge-m3 model automatski radi normalizaciju i pooling ako se koristi preko sentence-transformers
        # normalize_embeddings=True osigurava da su vektori spremni za cosine similarity
        vektori = model.encode(tekstovi, normalize_embeddings=True)
        
        # Pretvaramo numpy array u listu listi (kako Supabase očekuje)
        rezultat = vektori.tolist()
        
        logger.info(f"Lokalno generisano {len(rezultat)} embeddings (bge-m3).")
        return rezultat

    except Exception as e:
        logger.error(f"Greška pri lokalnom generisanju embeddinga: {e}")
        raise

def generiraj_jedan_embedding(model: SentenceTransformer, tekst: str) -> list[float]:
    """Generira embedding za jedan tekst."""
    return generiraj_embeddings(model, [tekst])[0]

def provjeri_hf_api() -> bool:
    """
    Budući da više ne koristimo API, ova funkcija sada samo potvrđuje 
    da je sistem spreman za lokalno procesuiranje.
    """
    return True