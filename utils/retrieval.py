# Verzija: 1.0 | Ažurirano: 2025-04-27

import logging
from sentence_transformers import SentenceTransformer
from chromadb import Collection

from config import RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD
from utils.embeddings import generiraj_jedan_embedding

logger = logging.getLogger(__name__)


def pretrazi(
    kolekcija: Collection,
    model: SentenceTransformer,
    upit: str,
    kategorija: str | None = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Pretražuje Chroma kolekciju i vraća relevantne chunkove.
    Opciono filtrira po kategoriji. Primjenjuje score threshold.
    Izlaz: lista dict-ova {tekst, score, metadata}
    """
    if not upit.strip():
        logger.warning("Prazan upit — preskačem pretragu.")
        return []

    # Generiraj embedding upita
    try:
        upit_embedding = generiraj_jedan_embedding(model, upit)
    except Exception as e:
        logger.error(f"Greška pri embedding upita: {e}")
        return []

    # Pripremi where filter
    where_filter = None
    if kategorija and kategorija != "Sve":
        where_filter = {"kategorija": kategorija}
        logger.info(f"Retrieval sa filterom kategorije: '{kategorija}'")
    else:
        logger.info("Retrieval bez filtera kategorije.")

    # Upit prema Chroma
    try:
        kwargs = {
            "query_embeddings": [upit_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        rezultati = kolekcija.query(**kwargs)

    except Exception as e:
        logger.error(f"Greška pri pretrazi Chroma: {e}")
        return []

    # Obradi rezultate
    chunkovi = []
    dokumenti = rezultati.get("documents", [[]])[0]
    metadati  = rezultati.get("metadatas", [[]])[0]
    distance  = rezultati.get("distances", [[]])[0]

    for tekst, meta, dist in zip(dokumenti, metadati, distance):
        # Chroma cosine distance: 0 = identično, 2 = suprotno
        # Konvertuj u similarity score (0.0 – 1.0)
        score = round(1 - (dist / 2), 4)

        if score < RETRIEVAL_SCORE_THRESHOLD:
            logger.info(
                f"Chunk iz '{meta.get('naziv_dokumenta')}' odbačen "
                f"(score={score} < threshold={RETRIEVAL_SCORE_THRESHOLD})."
            )
            continue

        chunkovi.append({
            "tekst":    tekst,
            "score":    score,
            "metadata": meta,
        })

    logger.info(f"Pronađeno {len(chunkovi)} relevantnih chunkova za upit.")
    return chunkovi


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
            f"[IZVOR {i}] {naziv} | {izvor} | {godina}\n"
            f"{chunk['tekst']}"
        )

    return "\n\n---\n\n".join(dijelovi)
