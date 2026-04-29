# Verzija: 1.0 | Ažurirano: 2025-04-27

import logging
import uuid
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb import Collection

from config import CHROMA_PATH, CHROMA_COLLECTION
from utils.embeddings import generiraj_embeddings

logger = logging.getLogger(__name__)


def kreiraj_klijent() -> chromadb.PersistentClient:
    """Kreira persistentni Chroma klijent."""
    try:
        klijent = chromadb.PersistentClient(path=str(CHROMA_PATH))
        logger.info(f"Chroma klijent kreiran: {CHROMA_PATH}")
        return klijent
    except Exception as e:
        logger.error(f"Greška pri kreiranju Chroma klijenta: {e}")
        raise


def dohvati_kolekciju(klijent: chromadb.PersistentClient) -> Collection:
    """Dohvata ili kreira Chroma kolekciju."""
    try:
        kolekcija = klijent.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        logger.info(f"Kolekcija '{CHROMA_COLLECTION}' dohvaćena ({kolekcija.count()} zapisa).")
        return kolekcija
    except Exception as e:
        logger.error(f"Greška pri dohvatanju kolekcije: {e}")
        raise


def dokument_postoji(kolekcija: Collection, naziv_dokumenta: str) -> bool:
    """Provjerava da li dokument već postoji u bazi po nazivu."""
    try:
        rezultat = kolekcija.get(where={"naziv_dokumenta": naziv_dokumenta}, limit=1)
        return len(rezultat["ids"]) > 0
    except Exception as e:
        logger.error(f"Greška pri provjeri postojanja dokumenta: {e}")
        return False


def dodaj_dokumente(
    kolekcija: Collection,
    model: SentenceTransformer,
    chunkovi: list[dict],
) -> int:
    """
    Dodaje chunkove u Chroma kolekciju.
    Vraća broj uspješno dodanih chunkova.
    """
    if not chunkovi:
        logger.warning("Nema chunkova za dodavanje.")
        return 0

    tekstovi = [c["tekst"] for c in chunkovi]
    metadati = [c["metadata"] for c in chunkovi]
    id_lista = [str(uuid.uuid4()) for _ in chunkovi]

    try:
        embeddings = generiraj_embeddings(model, tekstovi)

        kolekcija.add(
            ids=id_lista,
            embeddings=embeddings,
            documents=tekstovi,
            metadatas=metadati,
        )

        naziv = metadati[0].get("naziv_dokumenta", "nepoznat")
        logger.info(f"Dodano {len(chunkovi)} chunkova za dokument '{naziv}'.")
        return len(chunkovi)

    except Exception as e:
        logger.error(f"Greška pri dodavanju dokumenata u Chroma: {e}")
        return 0


def obrisi_dokument(kolekcija: Collection, naziv_dokumenta: str) -> bool:
    """Briše sve chunkove dokumenta iz kolekcije po nazivu fajla."""
    try:
        kolekcija.delete(where={"naziv_dokumenta": naziv_dokumenta})
        logger.info(f"Dokument '{naziv_dokumenta}' obrisan iz baze.")
        return True
    except Exception as e:
        logger.error(f"Greška pri brisanju dokumenta '{naziv_dokumenta}': {e}")
        return False


def lista_dokumenata(kolekcija: Collection) -> list[dict]:
    """
    Vraća listu jedinstvenih dokumenata u bazi sa brojem chunkova.
    Izlaz: [{'naziv_dokumenta': ..., 'kategorija': ..., 'godina': ..., 'chunkova': ...}]
    """
    try:
        svi = kolekcija.get(include=["metadatas"])
        if not svi["metadatas"]:
            return []

        # Grupiši po nazivu dokumenta
        dokumenti: dict[str, dict] = {}
        for meta in svi["metadatas"]:
            naziv = meta.get("naziv_dokumenta", "nepoznat")
            if naziv not in dokumenti:
                dokumenti[naziv] = {
                    "naziv_dokumenta": naziv,
                    "kategorija":      meta.get("kategorija", "—"),
                    "izvor":           meta.get("izvor", "—"),
                    "godina":          meta.get("godina", "—"),
                    "chunkova":        0,
                }
            dokumenti[naziv]["chunkova"] += 1

        return list(dokumenti.values())

    except Exception as e:
        logger.error(f"Greška pri dohvatanju liste dokumenata: {e}")
        return []


def ukupno_zapisa(kolekcija: Collection) -> int:
    """Vraća ukupan broj chunkova u kolekciji."""
    try:
        return kolekcija.count()
    except Exception as e:
        logger.error(f"Greška pri brojanju zapisa: {e}")
        return 0
