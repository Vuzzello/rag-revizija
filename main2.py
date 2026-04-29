# Verzija: 1.0 | Ažurirano: 2025-04-27

import logging
import tempfile
from pathlib import Path

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Your existing imports
from config import KATEGORIJE, PODRZANI_FORMATI
from config import KATEGORIJE, PODRZANI_FORMATI
from utils.embeddings import ucitaj_model
from utils.storage import (
    kreiraj_klijent,
    dohvati_kolekciju,
    dodaj_dokumente,
    obrisi_dokument,
    lista_dokumenata,
    dokument_postoji,
    ukupno_zapisa,
)
from utils.ingestion import ucitaj_dokument
from utils.chunking import podjeli_dokument
from utils.retrieval import pretrazi, formatiraj_kontekst
from utils.generation import provjeri_ollama, generiraj_odgovor_stream

logger = logging.getLogger(__name__)

# ── Stranica ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Interna Revizija",
    page_icon="📋",
    layout="wide",
)

# ── Inicijalizacija resursa (jednom pri pokretanju) ────────────────────────────
@st.cache_resource(show_spinner="Učitavanje embedding modela...")
def init_resursi():
    """Učitava embedding model i Chroma kolekciju jednom pri pokretanju."""
    model   = ucitaj_model()
    klijent = kreiraj_klijent()
    kol     = dohvati_kolekciju(klijent)
    return model, kol


model, kolekcija = init_resursi()

# ── Naslov ─────────────────────────────────────────────────────────────────────
st.title("📋 Baza znanja — Interna Revizija")
st.caption("Lokalni RAG sistem | Qwen3 via Ollama | BAAI/bge-m3 embeddings")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Status sistema")

    # Status embedding modela
    st.success("✅ Embedding model: Učitan")

    # Status Ollama
    if provjeri_ollama():
        st.success("✅ Ollama / Qwen3: Dostupan")
    else:
        st.error("❌ Ollama / Qwen3: Nedostupan — pokreni `ollama serve`")

    # Status baze
    broj_zapisa = ukupno_zapisa(kolekcija)
    dokumenti   = lista_dokumenata(kolekcija)
    st.info(f"🗄️ Baza: {len(dokumenti)} dokumenata | {broj_zapisa} chunkova")

    st.divider()

    # ── Upload dokumenata ──────────────────────────────────────────────────────
    st.header("📁 Upload dokumenata")

    uploadovani = st.file_uploader(
        "Odaberi fajlove (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    kategorija_upload = st.selectbox(
        "Kategorija dokumenata",
        options=KATEGORIJE,
        key="kategorija_upload",
    )
    izvor_upload = st.text_input("Izvor (npr. IIA, CBCG, Interna revizija)")
    godina_upload = st.text_input("Godina dokumenta (npr. 2024)")

    if st.button("📥 Indeksiraj dokumente", disabled=not uploadovani):
        if not izvor_upload or not godina_upload:
            st.warning("Upiši izvor i godinu prije indeksiranja.")
        else:
            metadata_base = {
                "kategorija": kategorija_upload,
                "izvor":      izvor_upload,
                "godina":     godina_upload,
            }

            progres = st.progress(0, text="Indeksiranje u toku...")
            ukupno  = len(uploadovani)

            for i, fajl in enumerate(uploadovani):
                sufiks = Path(fajl.name).suffix.lower()

                if sufiks not in PODRZANI_FORMATI:
                    st.warning(f"Format '{sufiks}' nije podržan: {fajl.name}")
                    continue

                # Provjeri duplikat
                if dokument_postoji(kolekcija, fajl.name):
                    st.warning(
                        f"'{fajl.name}' već postoji u bazi. "
                        f"Obriši ga ispod pa ponovo uploaduj."
                    )
                    continue

                # Sačuvaj privremeno na disk
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=sufiks
                ) as tmp:
                    tmp.write(fajl.read())
                    tmp_putanja = Path(tmp.name)

                # Ingestion → chunking → storage
                dokument = ucitaj_dokument(tmp_putanja, metadata_base)
                tmp_putanja.unlink(missing_ok=True)

                if not dokument:
                    st.error(f"Nije moguće parsirati '{fajl.name}'.")
                    continue

                # Postavi pravi naziv dokumenta
                dokument["metadata"]["naziv_dokumenta"] = fajl.name

                chunkovi = podjeli_dokument(dokument)
                dodano   = dodaj_dokumente(kolekcija, model, chunkovi)

                progres.progress(
                    (i + 1) / ukupno,
                    text=f"Obrađen: {fajl.name} ({dodano} chunkova)"
                )

            progres.empty()
            st.success("✅ Indeksiranje završeno!")
            st.rerun()

    st.divider()

    # ── Lista indeksiranih dokumenata ──────────────────────────────────────────
    st.header("📚 Indeksirani dokumenti")

    dokumenti = lista_dokumenata(kolekcija)

    if not dokumenti:
        st.caption("Nema indeksiranih dokumenata.")
    else:
        for dok in dokumenti:
            with st.expander(f"📄 {dok['naziv_dokumenta']}"):
                st.caption(f"Kategorija: **{dok['kategorija']}**")
                st.caption(f"Izvor: {dok['izvor']} | Godina: {dok['godina']}")
                st.caption(f"Chunkova: {dok['chunkova']}")
                if st.button(
                    "🗑️ Obriši",
                    key=f"obrisi_{dok['naziv_dokumenta']}",
                ):
                    obrisi_dokument(kolekcija, dok["naziv_dokumenta"])
                    st.success(f"'{dok['naziv_dokumenta']}' obrisan.")
                    st.rerun()

# ── Glavni panel ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col2:
    kategorija_upit = st.selectbox(
        "🔍 Filter kategorije",
        options=["Sve"] + KATEGORIJE,
        key="kategorija_upit",
    )

with col1:
    upit = st.text_area(
        "Unesite upit:",
        placeholder="npr. Koji su zahtjevi za upravljanje rizicima prema Basel III?",
        height=100,
    )

dugme = st.button(
    "🔎 Postavi pitanje",
    disabled=not upit.strip(),
    type="primary",
)

if dugme and upit.strip():
    if ukupno_zapisa(kolekcija) == 0:
        st.warning("Baza je prazna. Uploaduj i indeksiraj dokumente u sidebaru.")
    else:
        # Retrieval
        with st.spinner("Pretraživanje baze znanja..."):
            chunkovi = pretrazi(
                kolekcija,
                model,
                upit,
                kategorija=kategorija_upit,
            )

        if not chunkovi:
            st.warning(
                "Nije pronađen relevantan sadržaj u bazi znanja za ovaj upit. "
                "Pokušaj sa drugačijim pitanjem ili promijeni filter kategorije."
            )
        else:
            kontekst = formatiraj_kontekst(chunkovi)

            # Generacija odgovora (streaming)
            st.subheader("💬 Odgovor")
            with st.chat_message("assistant"):
                st.write_stream(
                    generiraj_odgovor_stream(upit, kontekst)
                )

            # Korišteni izvori
            with st.expander("📎 Korišteni izvori", expanded=True):
                zaglavlje = ["Dokument", "Kategorija", "Izvor", "Godina", "Relevantnost"]
                redovi = [
                    [
                        c["metadata"].get("naziv_dokumenta", "—"),
                        c["metadata"].get("kategorija", "—"),
                        c["metadata"].get("izvor", "—"),
                        c["metadata"].get("godina", "—"),
                        f"{round(c['score'] * 100, 1)}%",
                    ]
                    for c in chunkovi
                ]
                st.table([dict(zip(zaglavlje, r)) for r in redovi])

            # Detalji chunka
            with st.expander("🔬 Detalji chunka (za provjeru)"):
                for i, chunk in enumerate(chunkovi, 1):
                    st.markdown(f"**Chunk {i}** — score: `{chunk['score']}`")
                    st.text(chunk["tekst"][:500] + ("..." if len(chunk["tekst"]) > 500 else ""))
                    st.divider()