# Verzija: 2.0 | Cloud | Ažurirano: 2025-04-27

import logging
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

from config import KATEGORIJE, PODRZANI_FORMATI, provjeri_konfiguraciju
from utils.storage import (
    kreiraj_klijent,
    dodaj_dokumente,
    obrisi_dokument,
    lista_dokumenata,
    dokument_postoji,
    ukupno_zapisa,
)
from utils.ingestion import ucitaj_dokument
from utils.chunking import podjeli_dokument
from utils.retrieval import pretrazi, pretrazi_po_dokumentima, formatiraj_kontekst
from utils.generation import (
    provjeri_groq,
    generiraj_odgovor_stream,
    generiraj_compliance_stream,
)
from utils.embeddings import provjeri_hf_api

logger = logging.getLogger(__name__)

# ── Konfiguracija stranice ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Interna Revizija",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .compliance-header {
        background: linear-gradient(90deg, #1a3a5c, #2d6a9f);
        color: white; padding: 16px; border-radius: 8px; margin-bottom: 16px;
    }
    .status-ok  { color: #28a745; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Provjera konfiguracije ─────────────────────────────────────────────────────
nedostaje = provjeri_konfiguraciju()
if nedostaje:
    st.error(
        f"❌ Nedostaju API ključevi: **{', '.join(nedostaje)}**\n\n"
        f"Dodaj ih u Streamlit Cloud → Settings → Secrets."
    )
    st.stop()

# ── Inicijalizacija resursa ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Povezivanje sa bazom...")
def init_resursi():
    """Kreira Supabase klijent jednom pri pokretanju."""
    return kreiraj_klijent()


klijent_db = init_resursi()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 RAG — Interna Revizija")
    st.caption("Cloud verzija | Supabase + Groq + HuggingFace")

    st.divider()

    # Status
    st.subheader("⚙️ Status sistema")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # DB status (Supabase klijent je već inicijalizovan kao klijent_db)
        if klijent_db:
            st.success("DB")
        else:
            st.error("DB")
            
    with c2:
        # HF status
        if provjeri_hf_api():
            st.success("HF")
        else:
            st.error("HF")
            
    with c3:
        # LLM status (Groq)
        if provjeri_groq():
            st.success("LLM")
        else:
            st.error("LLM")

    # Status
    #st.subheader("⚙️ Status sistema")
    #c1, c2, c3 = st.columns(3)
    #with c1:
        #st.success("✅ DB")
    #with c2:
        #st.success("✅ HF") if provjeri_hf_api() else st.error("❌ HF")
    #with c3:
        #st.success("✅ LLM") if provjeri_groq() else st.error("❌ LLM")

    dokumenti_lista = lista_dokumenata(klijent_db)
    st.info(f"🗄️ **{len(dokumenti_lista)}** dok. | **{ukupno_zapisa(klijent_db)}** segm.")

    st.divider()

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.subheader("📁 Upload dokumenta")

    uploadovani = st.file_uploader(
        "PDF, DOCX ili TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    with st.expander("📝 Metapodaci", expanded=True):
        naziv_override    = st.text_input("Naziv", placeholder="Ostavi prazno = naziv fajla")
        kategorija_upload = st.selectbox("Kategorija *", KATEGORIJE)
        izvor_upload      = st.text_input("Izvor *", placeholder="npr. CBCG, IIA...")
        col_g, col_t      = st.columns(2)
        with col_g:
            godina_upload = st.text_input("Godina *", placeholder="2024", max_chars=4)
        with col_t:
            tip_dokumenta = st.selectbox("Tip", [
                "Zakon / Pravilnik", "Standard / Smjernica",
                "Procedura / Uputstvo", "Izvještaj",
                "Analiza / Ekspertiza", "Prezentacija", "Ostalo",
            ])
        napomena = st.text_area("Napomena", height=100)

    popunjeno = bool(izvor_upload.strip() and godina_upload.strip())
    if uploadovani and not popunjeno:
        st.warning("⚠️ Popuni Izvor i Godinu.")

    if st.button("📥 Indeksiraj", disabled=not (uploadovani and popunjeno),
                 type="primary", use_container_width=True):

        metadata_base = {
            "kategorija":    kategorija_upload,
            "izvor":         izvor_upload.strip(),
            "godina":        godina_upload.strip(),
            "tip_dokumenta": tip_dokumenta,
            "napomena":      napomena.strip(),
            "datum_uploada": datetime.now().strftime("%Y-%m-%d"),
        }

        progres   = st.progress(0)
        uspjesno  = 0

        for i, fajl in enumerate(uploadovani):
            sufiks    = Path(fajl.name).suffix.lower()
            naziv_dok = naziv_override.strip() if (naziv_override.strip() and i == 0) else fajl.name

            if sufiks not in PODRZANI_FORMATI:
                st.warning(f"⚠️ Nije podržan: {fajl.name}")
                continue

            if dokument_postoji(klijent_db, naziv_dok):
                st.warning(f"⚠️ Već postoji: '{naziv_dok}'")
                continue

            progres.progress((i + 0.3) / len(uploadovani), f"Parsiranje: {fajl.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=sufiks) as tmp:
                tmp.write(fajl.read())
                tmp_putanja = Path(tmp.name)

            meta_dok = {**metadata_base, "naziv_dokumenta": naziv_dok}
            dokument = ucitaj_dokument(tmp_putanja, meta_dok)
            tmp_putanja.unlink(missing_ok=True)

            if not dokument:
                st.error(f"❌ Nije moguće parsirati: {fajl.name}")
                continue

            dokument["metadata"]["naziv_dokumenta"] = naziv_dok
            progres.progress((i + 0.6) / len(uploadovani), f"Indeksiranje: {naziv_dok}")

            chunkovi = podjeli_dokument(dokument)
            dodano   = dodaj_dokumente(klijent_db, chunkovi)

            if dodano > 0:
                uspjesno += 1
                progres.progress((i + 1) / len(uploadovani), f"✅ {naziv_dok}")

        progres.empty()
        if uspjesno:
            st.success(f"✅ Indeksirano: {uspjesno} dokumenat(a)")
        st.rerun()


# ── Tabovi ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Pretraga baze znanja",
    "📚 Pregled dokumenata",
    "⚖️ Analiza usklađenosti",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRETRAGA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Pretraga baze znanja")

    col_f, col_q = st.columns([1, 3])
    with col_f:
        kategorija_upit = st.selectbox("Filter kategorije", ["Sve"] + KATEGORIJE)
    with col_q:
        upit = st.text_area(
            "Postavi pitanje:",
            placeholder="npr. Koji su zahtjevi za adekvatnost kapitala prema Basel III?",
            height=100,
        )

    if st.button("🔎 Pretraži", disabled=not upit.strip(), type="primary"):
        if ukupno_zapisa(klijent_db) == 0:
            st.warning("⚠️ Baza je prazna. Uploaduj dokumente.")
        else:
            with st.spinner("Pretraživanje..."):
                chunkovi = pretrazi(
                    klijent_db, upit,
                    kategorija=kategorija_upit if kategorija_upit != "Sve" else None,
                )

            if not chunkovi:
                st.warning("Nije pronađen relevantan sadržaj. Pokušaj drugačije pitanje.")
            else:
                kontekst = formatiraj_kontekst(chunkovi)
                st.subheader("💬 Odgovor")
                with st.chat_message("assistant", avatar="🤖"):
                    st.write_stream(generiraj_odgovor_stream(upit, kontekst))

                with st.expander(f"📎 Korišteni izvori ({len(chunkovi)})", expanded=True):
                    for chunk in chunkovi:
                        meta  = chunk["metadata"]
                        score = round(chunk["score"] * 100, 1)
                        ikona = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                        st.markdown(
                            f"{ikona} **{meta.get('naziv_dokumenta', '—')}** "
                            f"| `{meta.get('kategorija', '—')}` "
                            f"| {meta.get('izvor', '—')} "
                            f"| {meta.get('godina', '—')} "
                            f"| **{score}%**"
                        )

                with st.expander("🔬 Detalji segmenata"):
                    for i, chunk in enumerate(chunkovi, 1):
                        st.markdown(f"**Segment {i}** | score: `{chunk['score']}`")
                        st.text(chunk["tekst"][:400] + "..." if len(chunk["tekst"]) > 400 else chunk["tekst"])
                        st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREGLED DOKUMENATA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📚 Pregled indeksiranih dokumenata")

    dokumenti_lista = lista_dokumenata(klijent_db)

    if not dokumenti_lista:
        st.info("📭 Baza je prazna.")
    else:
        # Metrike
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dokumenata", len(dokumenti_lista))
        c2.metric("Segmenata", ukupno_zapisa(klijent_db))
        c3.metric("Kategorija", len(set(d["kategorija"] for d in dokumenti_lista)))
        avg_ch = round(sum(d["chunkova"] for d in dokumenti_lista) / len(dokumenti_lista), 1)
        c4.metric("Prosj. segm./dok.", avg_ch)

        st.divider()

        # Filteri
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            filt_kat = st.selectbox("Kategorija", ["Sve kategorije"] + KATEGORIJE)
        with cf2:
            godine = sorted(set(d["godina"] for d in dokumenti_lista if d.get("godina")), reverse=True)
            filt_god = st.selectbox("Godina", ["Sve godine"] + godine)
        with cf3:
            trazi = st.text_input("Pretraži naziv")

        filtrirani = dokumenti_lista
        if filt_kat != "Sve kategorije":
            filtrirani = [d for d in filtrirani if d["kategorija"] == filt_kat]
        if filt_god != "Sve godine":
            filtrirani = [d for d in filtrirani if d.get("godina") == filt_god]
        if trazi.strip():
            filtrirani = [d for d in filtrirani if trazi.lower() in d["naziv_dokumenta"].lower()]

        st.caption(f"Prikazano: **{len(filtrirani)}** od {len(dokumenti_lista)}")
        st.divider()

        # Prikaz po kategorijama
        for kat in KATEGORIJE:
            docs_u_kat = [d for d in filtrirani if d["kategorija"] == kat]
            if not docs_u_kat:
                continue
            st.subheader(f"📂 {kat} ({len(docs_u_kat)})")
            for dok in docs_u_kat:
                c1, c2, c3, c4, c5 = st.columns([3, 2, 1.5, 1.5, 1])
                c1.markdown(f"📄 **{dok['naziv_dokumenta']}**")
                c2.caption(f"🏛️ {dok.get('izvor','—')} | {dok.get('tip_dokumenta','—')}")
                c3.caption(f"📅 {dok.get('godina','—')}")
                c4.caption(f"📦 {dok['chunkova']} segm.")
                if c5.button("🗑️", key=f"del_{dok['naziv_dokumenta']}"):
                    obrisi_dokument(klijent_db, dok["naziv_dokumenta"])
                    st.success(f"✅ Obrisan: {dok['naziv_dokumenta']}")
                    st.rerun()
                st.divider()

        # CSV export
        if st.button("📊 Preuzmi listu kao CSV"):
            import csv, io
            out    = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=[
                "naziv_dokumenta", "kategorija", "izvor", "tip_dokumenta",
                "godina", "chunkova", "datum_uploada",
            ])
            writer.writeheader()
            for d in dokumenti_lista:
                writer.writerow({k: d.get(k, "") for k in writer.fieldnames})
            st.download_button(
                "⬇️ Preuzmi CSV",
                data=out.getvalue(),
                file_name=f"baza_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALIZA USKLAĐENOSTI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="compliance-header">
        <h2 style="margin:0">⚖️ Analiza usklađenosti dokumenta</h2>
        <p style="margin:4px 0 0 0; opacity:0.85">
            Uploaduj interni dokument i analiziraj usklađenost
            sa odabranim referentnim dokumentima iz baze znanja.
        </p>
    </div>
    """, unsafe_allow_html=True)

    dokumenti_za_ref = lista_dokumenata(klijent_db)
    if not dokumenti_za_ref:
        st.warning("⚠️ Baza znanja je prazna. Indeksiraj referentne dokumente prvo.")
    else:
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.subheader("📄 Interni dokument")
            interni_fajl = st.file_uploader(
                "Odaberi dokument za analizu",
                type=["pdf", "docx", "txt"],
                key="compliance_upload",
                help="Ovaj dokument se NE sprema u bazu — samo se analizira.",
            )
            if interni_fajl:
                st.success(f"✅ {interni_fajl.name}")

            naziv_int  = st.text_input("Naziv", value=interni_fajl.name if interni_fajl else "",
                                        placeholder="npr. Pravilnik o reviziji 2025")
            izvor_int  = st.text_input("Izvor", placeholder="npr. Interna izrada")
            godina_int = st.text_input("Godina", placeholder="2025")

        with col_r:
            st.subheader("📚 Referentni dokumenti")
            st.caption("Odaberi dokumente prema kojima se provjerava usklađenost:")

            odabrani: list[str] = []
            po_kat: dict[str, list] = {}
            for d in dokumenti_za_ref:
                po_kat.setdefault(d["kategorija"], []).append(d)

            for kat, docs in po_kat.items():
                st.markdown(f"**{kat}**")
                for d in docs:
                    if st.checkbox(
                        f"{d['naziv_dokumenta']} ({d.get('godina','—')})",
                        key=f"c_{d['naziv_dokumenta']}"
                    ):
                        odabrani.append(d["naziv_dokumenta"])

            if odabrani:
                st.info(f"✅ Odabrano: **{len(odabrani)}** referentnih dokumenata")

        st.divider()

        analiza_ok = bool(interni_fajl and odabrani and naziv_int.strip())
        if not analiza_ok:
            st.info("ℹ️ Potrebno: uploadovati dokument + unijeti naziv + odabrati barem 1 referentni.")

        if st.button("⚖️ Pokreni analizu usklađenosti",
                     disabled=not analiza_ok, type="primary", use_container_width=True):

            sufiks = Path(interni_fajl.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=sufiks) as tmp:
                tmp.write(interni_fajl.read())
                tmp_putanja = Path(tmp.name)

            meta_int = {
                "naziv_dokumenta": naziv_int.strip(),
                "kategorija":      "Interni",
                "izvor":           izvor_int.strip() or "Interna izrada",
                "godina":          godina_int.strip() or str(datetime.now().year),
            }

            # Parsiraj interni dokument
            dokument_int = ucitaj_dokument(tmp_putanja, meta_int)
            tmp_putanja.unlink(missing_ok=True)

            if not dokument_int:
                st.error("❌ Nije moguće parsirati interni dokument.")
            else:
                interni_tekst  = dokument_int["tekst"]
                chunkovi_int   = podjeli_dokument(dokument_int)

                # Dohvati referentne odlomke za prvih 5 chunkova
                svi_ref: list[dict] = []
                vidjeni: set[str]   = set()

                with st.spinner("🔍 Pretraživanje referentnih dokumenata..."):
                    za_pretragu = chunkovi_int[:7]  # Prvih 7 chunkova kao reprezentativni uzorak
                    for chunk in za_pretragu:
                        ref = pretrazi_po_dokumentima(
                            klijent_db, chunk["tekst"], odabrani, top_k=3
                        )
                        for r in ref:
                            kljuc = r["tekst"][:80]
                            if kljuc not in vidjeni:
                                vidjeni.add(kljuc)
                                svi_ref.append(r)

                ref_sortirani = sorted(svi_ref, key=lambda x: x["score"], reverse=True)[:10]
                ref_kontekst  = formatiraj_kontekst(ref_sortirani)

                # Prikaz parametara
                st.subheader("📊 Izvještaj o usklađenosti")
                ci1, ci2 = st.columns(2)
                ci1.markdown(f"**Analizirani:** `{naziv_int}`")
                ci2.markdown(
                    f"**Referentni ({len(odabrani)}):**  \n"
                    + "  \n".join(f"• `{n}`" for n in odabrani)
                )
                st.divider()

                # Streaming generacija
                placeholder   = st.empty()
                cijeli_tekst  = ""

                for token in generiraj_compliance_stream(
                    interni_tekst=interni_tekst[:4000],
                    referentni_kontekst=ref_kontekst,
                    naziv_internog=naziv_int.strip(),
                ):
                    cijeli_tekst += token
                    placeholder.markdown(cijeli_tekst + "▌")

                placeholder.markdown(cijeli_tekst)

                # Download izvještaja
                st.divider()
                naziv_fajla = (
                    f"uskladenost_{naziv_int.replace(' ', '_')}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}.md"
                )
                st.download_button(
                    "⬇️ Preuzmi izvještaj (Markdown)",
                    data=cijeli_tekst,
                    file_name=naziv_fajla,
                    mime="text/markdown",
                )
