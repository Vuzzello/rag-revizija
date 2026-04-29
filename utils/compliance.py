# Verzija: 1.0 | Ažurirano: 2025-04-27
# Modul za analizu usklađenosti internog dokumenta sa referentnim dokumentima

import logging
from typing import Generator

import ollama

from config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
)
from utils.embeddings import generiraj_embeddings, generiraj_jedan_embedding
from utils.chunking import podjeli_dokument
from utils.ingestion import ucitaj_dokument
from chromadb import Collection
from sentence_transformers import SentenceTransformer
from pathlib import Path

logger = logging.getLogger(__name__)

# ── System prompt za analizu usklađenosti ─────────────────────────────────────
COMPLIANCE_SYSTEM_PROMPT = """Ti si ekspert za pravnu i regulatornu usklađenost dokumenata u oblasti interne revizije.
Tvoj zadatak je da analiziraš interni dokument i pronađeš mjere u kojima nije usklađen sa referentnim dokumentima. 
Budi precizan i navodi konkretne odredbe. Ne izmišljaj reference koje nisu u kontekstu.

Tvoj odgovor MORA biti strukturiran tačno ovako:

## ✅ Usklađeni aspekti
[Navedi konkretne dijelove internog dokumenta koji su usklađeni sa referentnim dokumentima]

## ⚠️ Aspekti koji zahtijevaju usklađivanje
[Za svaki neusklađeni aspekt:]
**Problem:** [Opis konkretnog problema neusklađenosti]
**Referenca:** [Naziv referentnog dokumenta i relevantna odredba]
**Sugestija:** [Konkretan prijedlog kako uskladiti]

## 📋 Sažetak
**Ukupna procjena usklađenosti:** [Visoka / Srednja / Niska]
**Broj problema:** [broj]
**Prioritetne akcije:** [lista najvažnijih koraka za usklađivanje]

#"""


def dohvati_referentne_chunkove(
    kolekcija: Collection,
    model: SentenceTransformer,
    tekst_chunka: str,
    nazivi_dokumenata: list[str],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Za dati chunk internog dokumenta pretražuje referentne dokumente u bazi.
    Filtrira isključivo po odabranim referentnim dokumentima.
    """
    try:
        upit_embedding = generiraj_jedan_embedding(model, tekst_chunka)
    except Exception as e:
        logger.error(f"Greška pri generiranju embedding-a za compliance: {e}")
        return []

    # Chroma where filter — samo odabrani referentni dokumenti
    if len(nazivi_dokumenata) == 1:
        where_filter = {"naziv_dokumenta": nazivi_dokumenata[0]}
    else:
        where_filter = {"naziv_dokumenta": {"$in": nazivi_dokumenata}}

    try:
        rezultati = kolekcija.query(
            query_embeddings=[upit_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"Greška pri pretrazi referentnih dokumenata: {e}")
        return []

    chunkovi = []
    dokumenti = rezultati.get("documents", [[]])[0]
    metadati  = rezultati.get("metadatas", [[]])[0]
    distance  = rezultati.get("distances", [[]])[0]

    for tekst, meta, dist in zip(dokumenti, metadati, distance):
        score = round(1 - (dist / 2), 4)
        if score < RETRIEVAL_SCORE_THRESHOLD:
            continue
        chunkovi.append({"tekst": tekst, "score": score, "metadata": meta})

    return chunkovi


def formatiraj_compliance_prompt(
    interni_chunk: str,
    referentni_chunkovi: list[dict],
) -> str:
    """Kreira prompt za analizu usklađenosti jednog chunka."""
    referentni_tekst = ""
    for i, chunk in enumerate(referentni_chunkovi, 1):
        meta   = chunk["metadata"]
        naziv  = meta.get("naziv_dokumenta", "nepoznat")
        izvor  = meta.get("izvor", "—")
        godina = meta.get("godina", "—")
        referentni_tekst += (
            f"\n[REFERENCA {i}] {naziv} | {izvor} | {godina}\n"
            f"{chunk['tekst']}\n"
        )

    return (
        f"=== INTERNI DOKUMENT (odlomak za analizu) ===\n"
        f"{interni_chunk}\n\n"
        f"=== REFERENTNI DOKUMENTI ===\n"
        f"{referentni_tekst}\n\n"
        f"Analiziraj usklađenost internog dokumenta sa referentnim dokumentima."
    )


def analiziraj_uskladenost_stream(
    kolekcija: Collection,
    model: SentenceTransformer,
    putanja_dokumenta: Path,
    metadata_internog: dict,
    nazivi_referentnih: list[str],
) -> Generator[str, None, None]:
    """
    Glavni generator za analizu usklađenosti.
    Parsira interni dokument, za svaki chunk traži referentne odredbe,
    i generira strukturirani izvještaj.
    Yield-a tekst token po token.
    """
    # Parsiraj interni dokument
    dokument = ucitaj_dokument(putanja_dokumenta, metadata_internog)
    if not dokument:
        yield "❌ Nije moguće parsirati interni dokument."
        return

    chunkovi_internog = podjeli_dokument(dokument)
    if not chunkovi_internog:
        yield "❌ Interni dokument je prazan ili nije moguće podijeliti na segmente."
        return

    ukupno_chunkova = len(chunkovi_internog)
    logger.info(
        f"Analiza usklađenosti: {ukupno_chunkova} chunkova internog dokumenta "
        f"vs {len(nazivi_referentnih)} referentnih dokumenata."
    )

    # Sakupi sve relevantne referentne odredbe za cijeli dokument
    svi_referentni: list[dict] = []
    tekst_internog_kompletan = "\n\n".join(c["tekst"] for c in chunkovi_internog)

    # Za veće dokumente — uzorkuj reprezentativne chunkove
    if ukupno_chunkova > 10:
        # Uzmi prvih 5, zadnjih 3 i nekoliko iz sredine
        indeksi = (
            list(range(min(5, ukupno_chunkova))) +
            list(range(ukupno_chunkova // 3, ukupno_chunkova // 3 + 3)) +
            list(range(max(0, ukupno_chunkova - 3), ukupno_chunkova))
        )
        indeksi = sorted(set(indeksi))
        uzorkovani = [chunkovi_internog[i] for i in indeksi]
    else:
        uzorkovani = chunkovi_internog

    # Prikupi referentne chunkove za svaki odabrani chunk
    vidjeni_tekstovi: set[str] = set()
    for chunk in uzorkovani:
        ref_chunkovi = dohvati_referentne_chunkove(
            kolekcija, model, chunk["tekst"], nazivi_referentnih, top_k=3
        )
        for rc in ref_chunkovi:
            kljuc = rc["tekst"][:100]  # Izbjegni duplikate
            if kljuc not in vidjeni_tekstovi:
                vidjeni_tekstovi.add(kljuc)
                svi_referentni.append(rc)

    if not svi_referentni:
        yield (
            "⚠️ Nisu pronađeni relevantni odlomci u referentnim dokumentima.\n"
            "Provjeri da li su referentni dokumenti indeksirani u bazi znanja."
        )
        return

    # Ograniči broj referentnih chunkova za prompt (max 10)
    svi_referentni_sortirani = sorted(svi_referentni, key=lambda x: x["score"], reverse=True)
    top_referentni = svi_referentni_sortirani[:10]

    # Kreiraj finalni prompt
    prompt = (
        f"=== INTERNI DOKUMENT ZA ANALIZU ===\n"
        f"Naziv: {metadata_internog.get('naziv_dokumenta', 'nepoznat')}\n\n"
        f"{tekst_internog_kompletan[:4000]}"  # Ograniči na ~4000 znakova
        f"\n\n=== REFERENTNI DOKUMENTI ===\n"
    )

    for i, rc in enumerate(top_referentni, 1):
        meta   = rc["metadata"]
        naziv  = meta.get("naziv_dokumenta", "nepoznat")
        izvor  = meta.get("izvor", "—")
        godina = meta.get("godina", "—")
        prompt += (
            f"\n[REFERENCA {i}] {naziv} | {izvor} | {godina}\n"
            f"{rc['tekst']}\n"
        )

    prompt += "\n\nAnaliziraj usklađenost internog dokumenta sa referentnim dokumentima."

    # Generiraj odgovor putem Ollama
    try:
        klijent = ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)

        stream = klijent.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
            options={
                "temperature": 0.05,   # Niža temperatura = konzistentnije za pravne analize
                "num_predict": 2048,   # Duži odgovor za kompletan izvještaj
            },
        )

        for dio in stream:
            token = dio.message.content
            if token:
                yield token

    except ollama.ResponseError as e:
        logger.error(f"Ollama greška pri compliance analizi: {e}")
        yield f"\n\n❌ Greška pri generaciji izvještaja: {e}"
    except Exception as e:
        logger.error(f"Neočekivana greška pri compliance analizi: {e}")
        yield "\n\n❌ Desila se greška. Provjeri da li Ollama radi (`ollama serve`)."


def analiziraj_uskladenost(
    kolekcija: Collection,
    model: SentenceTransformer,
    putanja_dokumenta: Path,
    metadata_internog: dict,
    nazivi_referentnih: list[str],
) -> str:
    """Wrapper koji vraća kompletan string (bez streaminga) — za testiranje."""
    return "".join(analiziraj_uskladenost_stream(
        kolekcija, model, putanja_dokumenta, metadata_internog, nazivi_referentnih
    ))
