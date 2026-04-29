# Verzija: 1.0 | Ažurirano: 2025-04-27

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
    """
    Dijeli dokument na chunkove i svakom dodaje kompletan metadata.
    Ulaz: dict sa 'tekst' i 'metadata' ključevima.
    Izlaz: lista dict-ova sa 'tekst' i 'metadata' ključevima.
    """
    tekst: str   = dokument.get("tekst", "")
    metadata: dict = dokument.get("metadata", {})
    naziv: str   = metadata.get("naziv_dokumenta", "nepoznat")

    if not tekst.strip():
        logger.warning(f"Dokument '{naziv}' je prazan — preskačem chunking.")
        return []

    splitter = kreiraj_splitter()

    try:
        chunkovi_tekst: list[str] = splitter.split_text(tekst)
    except Exception as e:
        logger.error(f"Greška pri chunkingu dokumenta '{naziv}': {e}")
        return []

    ukupno = len(chunkovi_tekst)
    rezultat = []

    for index, chunk_tekst in enumerate(chunkovi_tekst):
        chunk_tekst = chunk_tekst.strip()
        if not chunk_tekst:
            continue

        rezultat.append({
            "tekst": chunk_tekst,
            "metadata": {
                **metadata,
                "chunk_index": index,
                "ukupno_chunkova": ukupno,
            },
        })

    logger.info(
        f"Dokument '{naziv}' podijeljen na {len(rezultat)} chunkova "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )

    return rezultat


def podjeli_vise_dokumenata(dokumenti: list[dict]) -> list[dict]:
    """Dijeli listu dokumenata na chunkove i vraća sve chunkove u jednoj listi."""
    svi_chunkovi = []

    for dokument in dokumenti:
        chunkovi = podjeli_dokument(dokument)
        svi_chunkovi.extend(chunkovi)

    logger.info(f"Ukupno kreirano {len(svi_chunkovi)} chunkova iz {len(dokumenti)} dokumenata.")
    return svi_chunkovi
