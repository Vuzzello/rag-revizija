# Verzija: 2.0 | Ažurirano: 2025-04-27

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS

logger = logging.getLogger(__name__)


def kreiraj_splitter() -> RecursiveCharacterTextSplitter:
    """Kreira i vraća konfigurisani text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def podjeli_dokument(dokument: dict) -> list[dict]:
    """Dijeli dokument na chunkove i svakom dodaje kompletan metadata."""
    tekst    = dokument.get("tekst", "")
    metadata = dokument.get("metadata", {})
    naziv    = metadata.get("naziv_dokumenta", "nepoznat")

    if not tekst.strip():
        logger.warning(f"Dokument '{naziv}' prazan — preskačem.")
        return []

    try:
        chunkovi_tekst = kreiraj_splitter().split_text(tekst)
    except Exception as e:
        logger.error(f"Greška pri chunkingu '{naziv}': {e}")
        return []

    ukupno   = len(chunkovi_tekst)
    rezultat = []
    for index, chunk_tekst in enumerate(chunkovi_tekst):
        chunk_tekst = chunk_tekst.strip()
        if not chunk_tekst:
            continue
        rezultat.append({
            "tekst": chunk_tekst,
            "metadata": {
                **metadata,
                "chunk_index":     index,
                "ukupno_chunkova": ukupno,
            },
        })

    logger.info(f"'{naziv}' → {len(rezultat)} chunkova.")
    return rezultat
