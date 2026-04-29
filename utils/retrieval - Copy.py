# Verzija: 2.0 | Ažurirano: 2025-04-27
# Retrieval modul — similarity pretraga putem Supabase pgvector

import logging
from supabase import Client

from config import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD
from utils.embeddings import generiraj_jedan_embedding

logger = logging.getLogger(__name__)


def pretrazi(
    klijent: Client,
    upit: str,
    kategorija: str | None = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Pretražuje Supabase pgvector kolekciju i vraća relevantne chunkove.
    Opciono filtrira po kategoriji. Primjenjuje score threshold.
    Izlaz: lista dict-ova {tekst, score, metadata}
    """
    if not upit.strip():
        logger.warning("Prazan upit.")
        return []

    try:
        upit_embedding = generiraj_jedan_embedding(upit)
    except Exception as e:
        logger.error(f"Greška pri generiranju embedding upita: {e}")
        return []

    try:
        rezultat = klijent.rpc(
            "pretrazi_dokumente",
            {
                "upit_embedding":    upit_embedding,
                "kategorija_filter": kategorija if kategorija and kategorija != "Sve" else None,
                "top_k":             top_k,
                "score_threshold":   RETRIEVAL_SCORE_THRESHOLD,
            },
        ).execute()

    except Exception as e:
        logger.error(f"Greška pri pretrazi Supabase: {e}")
        return []

    chunkovi = []
    for red in (rezultat.data or []):
        chunkovi.append({
            "tekst": red["tekst"],
            "score": round(float(red["similarity"]), 4),
            "metadata": {
                "naziv_dokumenta": red.get("naziv_dokumenta", "—"),
                "kategorija":      red.get("kategorija", "—"),
                "izvor":           red.get("izvor", "—"),
                "godina":          red.get("godina", "—"),
                "tip_dokumenta":   red.get("tip_dokumenta", "—"),
                "chunk_index":     red.get("chunk_index", 0),
                "ukupno_chunkova": red.get("ukupno_chunkova", 1),
            },
        })

    logger.info(f"Pronađeno {len(chunkovi)} relevantnih chunkova za upit.")
    return chunkovi


def pretrazi_po_dokumentima(
    klijent: Client,
    upit: str,
    nazivi_dokumenata: list[str],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Pretražuje samo unutar zadatih dokumenata (za compliance analizu).
    """
    if not upit.strip() or not nazivi_dokumenata:
        return []

    try:
        upit_embedding = generiraj_jedan_embedding(upit)
    except Exception as e:
        logger.error(f"Greška pri embedding upita: {e}")
        return []

    # Dohvati sve chunkove za odabrane dokumente
    try:
        rezultat = (
            klijent.table("dokumenti")
            .select("tekst, naziv_dokumenta, kategorija, izvor, godina, tip_dokumenta, chunk_index, ukupno_chunkova, embedding")
            .in_("naziv_dokumenta", nazivi_dokumenata)
            .execute()
        )
    except Exception as e:
        logger.error(f"Greška pri dohvatanju referentnih dokumenata: {e}")
        return []

    if not rezultat.data:
        return []

    # Izračunaj cosine similarity ručno
    def cosine_sim(a: list[float], b: list[float]) -> float:
        """Izračunava cosine similarity između dva vektora."""
        dot   = sum(x * y for x, y in zip(a, b))
        norma = (sum(x ** 2 for x in a) ** 0.5) * (sum(x ** 2 for x in b) ** 0.5)
        return dot / norma if norma > 0 else 0.0

    scorovani = []
    for red in rezultat.data:
        embedding_red = red.get("embedding")
        if not embedding_red:
            continue
        score = cosine_sim(upit_embedding, embedding_red)
        if score >= RETRIEVAL_SCORE_THRESHOLD:
            scorovani.append({
                "tekst": red["tekst"],
                "score": round(score, 4),
                "metadata": {
                    "naziv_dokumenta": red.get("naziv_dokumenta", "—"),
                    "kategorija":      red.get("kategorija", "—"),
                    "izvor":           red.get("izvor", "—"),
                    "godina":          red.get("godina", "—"),
                    "tip_dokumenta":   red.get("tip_dokumenta", "—"),
                    "chunk_index":     red.get("chunk_index", 0),
                    "ukupno_chunkova": red.get("ukupno_chunkova", 1),
                },
            })

    # Sortiraj po score-u i vrati top_k
    scorovani.sort(key=lambda x: x["score"], reverse=True)
    return scorovani[:top_k]


def formatiraj_kontekst(chunkovi: list[dict]) -> str:
    """Formatira chunkove u kontekst string za LLM prompt."""
    if not chunkovi:
        return ""

    dijelovi = []
    for i, chunk in enumerate(chunkovi, 1):
        meta   = chunk["metadata"]
        naziv  = meta.get("naziv_dokumenta", "nepoznat")
        godina = meta.get("godina", "—")
        izvor  = meta.get("izvor", "—")
        dijelovi.append(
            f"[IZVOR {i}] {naziv} | {izvor} | {godina}\n{chunk['tekst']}"
        )
    return "\n\n---\n\n".join(dijelovi)
