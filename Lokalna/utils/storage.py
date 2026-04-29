import logging
import uuid
import streamlit as st
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from utils.embeddings import generiraj_embeddings

logger = logging.getLogger(__name__)

def kreiraj_klijent() -> Client:
    """Kreira klijent za povezivanje sa Supabase bazi."""
    try:
        # Koristi ključeve koje ste već unijeli u Streamlit Secrets
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Greška pri povezivanju na Supabase: {e}")
        raise

def dodaj_dokumente(
    klijent: Client,
    model: SentenceTransformer,
    chunkovi: list[dict],
) -> int:
    """Dodaje chunkove u Supabase tabelu 'dokumenti'."""
    if not chunkovi:
        return 0

    tekstovi = [c["tekst"] for c in chunkovi]
    embeddings = generiraj_embeddings(model, tekstovi)

    podaci_za_unos = []
    for i, c in enumerate(chunkovi):
        podaci_za_unos.append({
            "tekst": c["tekst"],
            "embedding": embeddings[i].tolist(),
            "naziv_dokumenta": c["metadata"].get("naziv_dokumenta"),
            "kategorija": c["metadata"].get("kategorija", "—"),
            "izvor": c["metadata"].get("izvor", "—"),
            "godina": str(c["metadata"].get("godina", "—")),
            "metadata": c["metadata"]
        })

    try:
        # Naziv tabele je 'dokumenti' prema vašem SQL setupu
        rezultat = klijent.table("dokumenti").insert(podaci_za_unos).execute()
        logger.info(f"Uspješno dodano {len(podaci_za_unos)} chunkova u Supabase.")
        return len(podaci_za_unos)
    except Exception as e:
        logger.error(f"Greška pri upisu u Supabase: {e}")
        return 0

def dokument_postoji(klijent: Client, naziv_dokumenta: str) -> bool:
    """Provjerava da li dokument već postoji u bazi."""
    try:
        rezultat = klijent.table("dokumenti").select("id").eq("naziv_dokumenta", naziv_dokumenta).limit(1).execute()
        return len(rezultat.data) > 0
    except Exception:
        return False

def obrisi_dokument(klijent: Client, naziv_dokumenta: str) -> bool:
    """Briše dokument iz baze."""
    try:
        klijent.table("dokumenti").delete().eq("naziv_dokumenta", naziv_dokumenta).execute()
        return True
    except Exception as e:
        logger.error(f"Greška pri brisanju dokumenta: {e}")
        return False

def lista_dokumenata(klijent: Client) -> list[dict]:
    """Vraća listu dokumenata iz baze."""
    try:
        rezultat = klijent.table("dokumenti").select("naziv_dokumenta, kategorija, izvor, godina").execute()
        
        dokumenti = {}
        for r in rezultat.data:
            naziv = r["naziv_dokumenta"]
            if naziv not in dokumenti:
                dokumenti[naziv] = {
                    "naziv_dokumenta": naziv,
                    "kategorija": r["kategorija"],
                    "izvor": r["izvor"],
                    "godina": r["godina"],
                    "chunkova": 0
                }
            dokumenti[naziv]["chunkova"] += 1
            
        return list(dokumenti.values())
    except Exception as e:
        logger.error(f"Greška pri dohvatanju liste dokumenata: {e}")
        return []

def ukupno_zapisa(klijent: Client) -> int:
    """Vraća ukupan broj zapisa u bazi."""
    try:
        rezultat = klijent.table("dokumenti").select("id", count="exact").execute()
        return rezultat.count if rezultat.count else 0
    except Exception:
        return 0