# Verzija: 2.0 | Ažurirano: 2025-04-27
# Embeddings putem HuggingFace Inference API (BAAI/bge-m3)

import logging
import time
import httpx

from config import HF_API_TOKEN, HF_API_URL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# HF API zaglavlja
_HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}


def _normalizuj_vektor(vektor: list[float]) -> list[float]:
    """Normalizuje vektor na jediničnu dužinu (za cosine similarity)."""
    norma = sum(x ** 2 for x in vektor) ** 0.5
    if norma == 0:
        return vektor
    return [x / norma for x in vektor]


def generiraj_embeddings(
    tekstovi: list[str],
    max_pokusaja: int = 3,
    pauza_sekundi: int = 20,
) -> list[list[float]]:
    """
    Generira embeddings za listu tekstova putem HuggingFace Inference API.
    Automatski pokušava ponovo ako je model u 'loading' stanju.
    """
    if not tekstovi:
        logger.warning("Proslijeđena prazna lista tekstova.")
        return []

    for pokusaj in range(max_pokusaja):
        try:
            odgovor = httpx.post(
                HF_API_URL,
                headers=_HEADERS,
                json={
                    "inputs": tekstovi,
                    "options": {"wait_for_model": True},
                },
                timeout=60.0,
            )

            if odgovor.status_code == 503:
                # Model se učitava — čekaj i pokušaj ponovo
                logger.warning(
                    f"HF model se učitava (pokušaj {pokusaj + 1}/{max_pokusaja}). "
                    f"Čekanje {pauza_sekundi}s..."
                )
                time.sleep(pauza_sekundi)
                continue

            odgovor.raise_for_status()
            vektori = odgovor.json()

            # HF vraća list[list[float]] ili list[list[list[float]]] za bge-m3
            # bge-m3 vraća [batch, seq_len, dim] — uzimamo mean pooling
            rezultat = []
            for v in vektori:
                if isinstance(v[0], list):
                    # Mean pooling po seq_len dimenziji
                    dim = len(v[0])
                    pooled = [
                        sum(token[d] for token in v) / len(v)
                        for d in range(dim)
                    ]
                    rezultat.append(_normalizuj_vektor(pooled))
                else:
                    rezultat.append(_normalizuj_vektor(v))

            logger.info(f"Generisano {len(rezultat)} embeddings putem HF API.")
            return rezultat

        except httpx.HTTPStatusError as e:
            logger.error(f"HF API HTTP greška: {e.response.status_code} — {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Greška pri pozivu HF API: {e}")
            if pokusaj < max_pokusaja - 1:
                time.sleep(5)
                continue
            raise

    raise RuntimeError(f"HF API nije dostupan nakon {max_pokusaja} pokušaja.")


def generiraj_jedan_embedding(tekst: str) -> list[float]:
    """Generira embedding za jedan tekst."""
    return generiraj_embeddings([tekst])[0]


def provjeri_hf_api() -> bool:
    """Provjerava dostupnost HuggingFace Inference API."""
    try:
        rezultat = generiraj_embeddings(["test"])
        return len(rezultat) > 0
    except Exception as e:
        logger.error(f"HF API nije dostupan: {e}")
        return False
