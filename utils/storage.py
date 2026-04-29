# Verzija: 2.0 | Ažurirano: 2025-04-27
# Storage modul — Supabase pgvector umjesto ChromaDB

import logging
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY
from utils.embeddings import generiraj_embeddings

logger = logging.getLogger(__name__)


def kreiraj_klijent() -> Client:
    """Kreira i vraća Supabase klijent."""
    try:
        klijent = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase klijent kreiran.")
        return klijent
    except Exception as e:
        logger.error(f"Greška pri kreiranju Supabase klijenta: {e}")
        raise


def dokument_postoji(klijent: Client, naziv_dokumenta: str) -> bool:
    """Provjerava da li dokument već postoji u bazi po nazivu."""
    try:
        rezultat = (
            klijent.table("dokumenti")
            .select("id", count="exact")
            .eq("naziv_dokumenta", naziv_dokumenta)
            .limit(1)
            .execute()
        )
        return (rezultat.count or 0) > 0
    except Exception as e:
        logger.error(f"Greška pri provjeri duplikata: {e}")
        return False


def dodaj_dokumente(klijent: Client, chunkovi: list[dict]) -> int:
    """
    Dodaje chunkove u Supabase tabelu 'dokumenti'.
    Vraća broj uspješno dodanih chunkova.
    """
    if not chunkovi:
        logger.warning("Nema chunkova za dodavanje.")
        return 0

    tekstovi = [c["tekst"] for c in chunkovi]
    metadati = [c["metadata"] for c in chunkovi]

    try:
        # Generiraj embeddings za sve chunkove
        embeddings = generiraj_embeddings(tekstovi)

        # Pripremi redove za insert
        redovi = []
        for tekst, meta, embedding in zip(tekstovi, metadati, embeddings):
            redovi.append({
                "naziv_dokumenta": meta.get("naziv_dokumenta", "nepoznat"),
                "kategorija":      meta.get("kategorija", "Ostali"),
                "izvor":           meta.get("izvor", ""),
                "godina":          meta.get("godina", ""),
                "tip_dokumenta":   meta.get("tip_dokumenta", ""),
                "napomena":        meta.get("napomena", ""),
                "datum_uploada":   meta.get("datum_uploada", ""),
                "chunk_index":     meta.get("chunk_index", 0),
                "ukupno_chunkova": meta.get("ukupno_chunkova", 1),
                "tekst":           tekst,
                "embedding":       embedding,
            })

        # Batch insert (Supabase prima max 1000 redova odjednom)
        velicina_batcha = 100
        ukupno_dodano = 0

        for i in range(0, len(redovi), velicina_batcha):
            batch = redovi[i: i + velicina_batcha]
            klijent.table("dokumenti").insert(batch).execute()
            ukupno_dodano += len(batch)
            logger.info(f"Dodan batch {i // velicina_batcha + 1}: {len(batch)} chunkova")

        naziv = metadati[0].get("naziv_dokumenta", "nepoznat")
        logger.info(f"Ukupno dodano {ukupno_dodano} chunkova za '{naziv}'.")
        return ukupno_dodano

    except Exception as e:
        logger.error(f"Greška pri dodavanju dokumenata u Supabase: {e}")
        return 0


def obrisi_dokument(klijent: Client, naziv_dokumenta: str) -> bool:
    """Briše sve chunkove dokumenta iz Supabase tabele."""
    try:
        klijent.table("dokumenti").delete().eq("naziv_dokumenta", naziv_dokumenta).execute()
        logger.info(f"Dokument '{naziv_dokumenta}' obrisan.")
        return True
    except Exception as e:
        logger.error(f"Greška pri brisanju dokumenta '{naziv_dokumenta}': {e}")
        return False


def lista_dokumenata(klijent: Client) -> list[dict]:
    """
    Vraća listu jedinstvenih dokumenata sa brojem chunkova.
    Koristi SQL funkciju lista_dokumenata_unique() iz Supabase.
    """
    try:
        rezultat = klijent.rpc("lista_dokumenata_unique").execute()
        return rezultat.data or []
    except Exception as e:
        logger.error(f"Greška pri dohvatanju liste dokumenata: {e}")
        return []


def ukupno_zapisa(klijent: Client) -> int:
    """Vraća ukupan broj chunkova u tabeli."""
    try:
        rezultat = klijent.table("dokumenti").select("id", count="exact").execute()
        return rezultat.count or 0
    except Exception as e:
        logger.error(f"Greška pri brojanju zapisa: {e}")
        return 0
