# Verzija: 2.0 | Ažurirano: 2025-04-27
# Glavni Streamlit UI — RAG baza znanja za interne revizore

import logging
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

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
from utils.compliance import analiziraj_uskladenost_stream

logger = logging.getLogger(__name__)

# ── Konfiguracija stranice ────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Interna Revizija",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS stilovi ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .status-box {
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.9em;
    }
    .doc-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .kategorija-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
        background-color: #4a90d9;
    }
    .compliance-header {
        background: linear-gradient(90deg, #1a3a5c, #2d6a9f);
        color: white;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Inicijalizacija resursa (jednom pri pokretanju) ───────────────────────────
@st.cache_resource(show_spinner="⏳ Učitavanje sistema...")
def init_resursi():
    """Učitava embedding model i Chroma kolekciju jednom pri pokretanju."""
    model   = ucitaj_model()
    klijent = kreiraj_klijent()
    kol     = dohvati_kolekciju(klijent)
    return model, kol


model, kolekcija = init_resursi()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2171/2171998.png", width=60)
    st.title("RAG — Interna Revizija")
    st.caption("Lokalni AI sistem | v2.0")

    st.divider()

    # ── Status sistema ─────────────────────────────────────────────────────────
    st.subheader("⚙️ Status sistema")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.success("✅ Embeddings")
    with col_s2:
        if provjeri_ollama():
            st.success("✅ Ollama")
        else:
            st.error("❌ Ollama")
            st.caption("Pokreni: `ollama serve`")

    dokumenti_lista = lista_dokumenata(kolekcija)
    broj_dok        = len(dokumenti_lista)
    broj_chunkova   = ukupno_zapisa(kolekcija)
    st.info(f"🗄️ **{broj_dok}** dokumenata | **{broj_chunkova}** segmenata")

    st.divider()

    # ── Upload dokumenata ──────────────────────────────────────────────────────
    st.subheader("📁 Upload dokumenta")

    uploadovani = st.file_uploader(
        "Odaberi fajlove (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Možeš odabrati više fajlova odjednom. Svaki fajl dobija iste metapodatke.",
    )

    with st.expander("📝 Metapodaci dokumenta", expanded=True):

        naziv_override = st.text_input(
            "Naziv dokumenta",
            placeholder="Ostavi prazno = koristi naziv fajla",
            help="Ako uploduješ više fajlova, ovo polje se primjenjuje samo na prvi fajl.",
        )

        kategorija_upload = st.selectbox(
            "Kategorija *",
            options=KATEGORIJE,
            help="Obavezno polje. Koristi se za filtriranje pri pretrazi.",
        )

        izvor_upload = st.text_input(
            "Izvor *",
            placeholder="npr. CBCG, IIA, Basel odbor, Interna izrada...",
            help="Ko je izdao ili kreirao dokument.",
        )

        col_g, col_t = st.columns(2)
        with col_g:
            godina_upload = st.text_input(
                "Godina *",
                placeholder="npr. 2024",
                max_chars=4,
            )
        with col_t:
            tip_dokumenta = st.selectbox(
                "Tip",
                options=[
                    "Zakon / Pravilnik",
                    "Standard / Smjernica",
                    "Procedura / Uputstvo",
                    "Izvještaj",
                    "Analiza / Ekspertiza",
                    "Prezentacija",
                    "Ostalo",
                ],
            )

        napomena = st.text_area(
            "Napomena (opcionalno)",
            placeholder="Kratki opis sadržaja...",
            height=60,
        )

    # Validacija prije indeksiranja
    obavezna_polja_popunjena = bool(izvor_upload and godina_upload)

    if not obavezna_polja_popunjena and uploadovani:
        st.warning("⚠️ Popuni sva obavezna polja (*) prije indeksiranja.")

    if st.button(
        "📥 Indeksiraj dokumente",
        disabled=not (uploadovani and obavezna_polja_popunjena),
        type="primary",
        use_container_width=True,
    ):
        metadata_base = {
            "kategorija":     kategorija_upload,
            "izvor":          izvor_upload,
            "godina":         godina_upload,
            "tip_dokumenta":  tip_dokumenta,
            "napomena":       napomena,
            "datum_uploada":  datetime.now().strftime("%Y-%m-%d"),
        }

        progres   = st.progress(0, text="Priprema za indeksiranje...")
        ukupno    = len(uploadovani)
        uspjesno  = 0
        neuspjesno = 0

        for i, fajl in enumerate(uploadovani):
            sufiks = Path(fajl.name).suffix.lower()

            if sufiks not in PODRZANI_FORMATI:
                st.warning(f"⚠️ Format '{sufiks}' nije podržan: {fajl.name}")
                neuspjesno += 1
                continue

            # Naziv dokumenta: override ili originalni naziv fajla
            naziv_dok = naziv_override.strip() if (naziv_override.strip() and i == 0) else fajl.name

            # Provjeri duplikat
            if dokument_postoji(kolekcija, naziv_dok):
                st.warning(f"⚠️ '{naziv_dok}' već postoji. Obriši ga u listi, pa ponovo uploaduj.")
                neuspjesno += 1
                continue

            progres.progress((i + 0.3) / ukupno, text=f"Parsiranje: {fajl.name}")

            # Sačuvaj privremeno
            with tempfile.NamedTemporaryFile(delete=False, suffix=sufiks) as tmp:
                tmp.write(fajl.read())
                tmp_putanja = Path(tmp.name)

            # Ingestion
            meta_dok = {**metadata_base, "naziv_dokumenta": naziv_dok}
            dokument = ucitaj_dokument(tmp_putanja, meta_dok)
            tmp_putanja.unlink(missing_ok=True)

            if not dokument:
                st.error(f"❌ Nije moguće parsirati '{fajl.name}'.")
                neuspjesno += 1
                continue

            dokument["metadata"]["naziv_dokumenta"] = naziv_dok

            progres.progress((i + 0.6) / ukupno, text=f"Indeksiranje: {naziv_dok}")

            # Chunking + storage
            chunkovi = podjeli_dokument(dokument)
            dodano   = dodaj_dokumente(kolekcija, model, chunkovi)

            if dodano > 0:
                uspjesno += 1
                progres.progress((i + 1) / ukupno, text=f"✅ {naziv_dok} ({dodano} segmenata)")
            else:
                neuspjesno += 1

        progres.empty()

        if uspjesno > 0:
            st.success(f"✅ Indeksirano: {uspjesno} dokumenat(a)")
        if neuspjesno > 0:
            st.error(f"❌ Neuspješno: {neuspjesno} dokumenat(a)")

        st.rerun()


# ── Glavni sadržaj — tabovi ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📚 Pregled dokumenata", 
    "🔍 Pretraga baze znanja",
    "⚖️ Analiza usklađenosti",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRETRAGA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Pretraga baze znanja")

    col_f, col_q = st.columns([1, 3])

    with col_f:
        kategorija_upit = st.selectbox(
            "Filter kategorije",
            options=["Sve"] + KATEGORIJE,
            key="kategorija_upit",
        )
        tip_filter = st.selectbox(
            "Filter tipa",
            options=["Sve vrste", "Zakon / Pravilnik", "Standard / Smjernica",
                     "Procedura / Uputstvo", "Izvještaj", "Analiza / Ekspertiza",
                     "Prezentacija", "Ostalo"],
            key="tip_filter",
        )

    with col_q:
        upit = st.text_area(
            "Postavi pitanje:",
            placeholder="npr. Koji su minimalni zahtjevi za adekvatnost kapitala prema Basel III?",
            height=100,
            key="upit_pretraga",
        )

    dugme_pretraga = st.button(
        "🔎 Pretraži",
        disabled=not upit.strip(),
        type="primary",
        key="btn_pretraga",
    )

    if dugme_pretraga and upit.strip():
        if ukupno_zapisa(kolekcija) == 0:
            st.warning("⚠️ Baza je prazna. Uploaduj i indeksiraj dokumente u sidebaru.")
        else:
            with st.spinner("🔍 Pretraživanje baze znanja..."):
                chunkovi = pretrazi(
                    kolekcija,
                    model,
                    upit,
                    kategorija=kategorija_upit if kategorija_upit != "Sve" else None,
                )

            if not chunkovi:
                st.warning(
                    "Nije pronađen relevantan sadržaj za ovaj upit. "
                    "Pokušaj sa drugačijim pitanjem ili proširi filter kategorije."
                )
            else:
                kontekst = formatiraj_kontekst(chunkovi)

                st.subheader("💬 Odgovor")
                with st.chat_message("assistant", avatar="🤖"):
                    st.write_stream(generiraj_odgovor_stream(upit, kontekst))

                col_i1, col_i2 = st.columns(2)

                with col_i1:
                    with st.expander(f"📎 Korišteni izvori ({len(chunkovi)})", expanded=True):
                        for chunk in chunkovi:
                            meta = chunk["metadata"]
                            score_pct = round(chunk["score"] * 100, 1)
                            score_color = (
                                "🟢" if score_pct >= 70
                                else "🟡" if score_pct >= 50
                                else "🔴"
                            )
                            st.markdown(
                                f"{score_color} **{meta.get('naziv_dokumenta', '—')}**  \n"
                                f"Kategorija: `{meta.get('kategorija', '—')}` | "
                                f"Izvor: {meta.get('izvor', '—')} | "
                                f"Godina: {meta.get('godina', '—')} | "
                                f"Relevantnost: **{score_pct}%**"
                            )
                            st.divider()

                with col_i2:
                    with st.expander("🔬 Detalji segmenata (za provjeru)"):
                        for i, chunk in enumerate(chunkovi, 1):
                            st.markdown(
                                f"**Segment {i}** | "
                                f"score: `{chunk['score']}` | "
                                f"{chunk['metadata'].get('naziv_dokumenta', '—')} "
                                f"(chunk {chunk['metadata'].get('chunk_index', '?')}/"
                                f"{chunk['metadata'].get('ukupno_chunkova', '?')})"
                            )
                            st.text(chunk["tekst"][:400] + ("..." if len(chunk["tekst"]) > 400 else ""))
                            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREGLED DOKUMENATA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📚 Pregled indeksiranih dokumenata")

    dokumenti_lista = lista_dokumenata(kolekcija)

    if not dokumenti_lista:
        st.info("📭 Baza je prazna. Uploaduj dokumente putem sidebara.")
    else:
        # ── Statistike ─────────────────────────────────────────────────────────
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Ukupno dokumenata", len(dokumenti_lista))
        with col_m2:
            st.metric("Ukupno segmenata", ukupno_zapisa(kolekcija))
        with col_m3:
            kategorije_u_bazi = set(d["kategorija"] for d in dokumenti_lista)
            st.metric("Aktivnih kategorija", len(kategorije_u_bazi))
        with col_m4:
            prosj_chunkova = round(
                sum(d["chunkova"] for d in dokumenti_lista) / len(dokumenti_lista), 1
            )
            st.metric("Prosj. segmenata/dok.", prosj_chunkova)

        st.divider()

        # ── Filter i pretraga ──────────────────────────────────────────────────
        col_fil1, col_fil2, col_fil3 = st.columns(3)

        with col_fil1:
            filter_kat = st.selectbox(
                "Filtriraj po kategoriji",
                options=["Sve kategorije"] + KATEGORIJE,
                key="filter_kat_pregled",
            )
        with col_fil2:
            filter_god = st.selectbox(
                "Filtriraj po godini",
                options=["Sve godine"] + sorted(
                    set(d["godina"] for d in dokumenti_lista if d["godina"] != "—"),
                    reverse=True,
                ),
                key="filter_god_pregled",
            )
        with col_fil3:
            pretrazi_naziv = st.text_input(
                "Pretraži po nazivu",
                placeholder="Ukucaj dio naziva...",
                key="pretrazi_naziv",
            )

        # Primijeni filtere
        filtrirani = dokumenti_lista
        if filter_kat != "Sve kategorije":
            filtrirani = [d for d in filtrirani if d["kategorija"] == filter_kat]
        if filter_god != "Sve godine":
            filtrirani = [d for d in filtrirani if d["godina"] == filter_god]
        if pretrazi_naziv.strip():
            filtrirani = [
                d for d in filtrirani
                if pretrazi_naziv.lower() in d["naziv_dokumenta"].lower()
            ]

        st.caption(f"Prikazano: {len(filtrirani)} od {len(dokumenti_lista)} dokumenata")

        st.divider()

        # ── Grupisanje po kategorijama ─────────────────────────────────────────
        kategorije_redosljed = KATEGORIJE + ["—"]  # Osiguraj redosljed

        for kategorija in kategorije_redosljed:
            docs_u_kat = [d for d in filtrirani if d["kategorija"] == kategorija]
            if not docs_u_kat:
                continue

            # Zaglavlje kategorije
            st.subheader(f"📂 {kategorija} ({len(docs_u_kat)})")

            # Tabela za kategoriju
            for dok in docs_u_kat:
                with st.container():
                    col_d1, col_d2, col_d3, col_d4, col_d5, col_d6 = st.columns(
                        [3, 2, 1.5, 1.5, 1, 1]
                    )

                    with col_d1:
                        st.markdown(f"📄 **{dok['naziv_dokumenta']}**")
                        if dok.get("napomena"):
                            st.caption(dok["napomena"])

                    with col_d2:
                        st.caption(f"🏛️ {dok.get('izvor', '—')}")
                        st.caption(f"📋 {dok.get('tip_dokumenta', '—')}")

                    with col_d3:
                        st.caption(f"📅 {dok.get('godina', '—')}")

                    with col_d4:
                        st.caption(f"📦 {dok['chunkova']} segmenata")

                    with col_d5:
                        st.caption(f"📆 {dok.get('datum_uploada', '—')}")

                    with col_d6:
                        if st.button(
                            "🗑️",
                            key=f"obrisi_{dok['naziv_dokumenta']}",
                            help=f"Obriši '{dok['naziv_dokumenta']}' iz baze",
                        ):
                            obrisi_dokument(kolekcija, dok["naziv_dokumenta"])
                            st.success(f"✅ '{dok['naziv_dokumenta']}' obrisan.")
                            st.rerun()

                    st.divider()

        # ── Export liste ───────────────────────────────────────────────────────
        if st.button("📊 Preuzmi listu kao CSV", key="export_csv"):
            import csv, io
            output = io.StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "naziv_dokumenta", "kategorija", "izvor", "tip_dokumenta",
                    "godina", "chunkova", "datum_uploada", "napomena"
                ]
            )
            writer.writeheader()
            for d in dokumenti_lista:
                writer.writerow(d)
            st.download_button(
                label="⬇️ Preuzmi CSV",
                data=output.getvalue(),
                file_name=f"baza_dokumenata_{datetime.now().strftime('%Y%m%d')}.csv",
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
            Uploaduj interni dokument i analiziraj njegovu usklađenost
            sa referentnim dokumentima iz baze znanja.
        </p>
    </div>
    """, unsafe_allow_html=True)

    dokumenti_lista_compliance = lista_dokumenata(kolekcija)

    if not dokumenti_lista_compliance:
        st.warning(
            "⚠️ Baza znanja je prazna. Potrebno je prvo indeksirati referentne dokumente "
            "(zakoni, standardi, procedure) putem sidebara."
        )
    else:
        col_l, col_r = st.columns([1, 1])

        # ── Lijeva kolona: Upload internog dokumenta ───────────────────────────
        with col_l:
            st.subheader("📄 Interni dokument (za analizu)")

            interni_fajl = st.file_uploader(
                "Uploaduj interni dokument",
                type=["pdf", "docx", "txt"],
                key="interni_dokument",
                help="Ovaj dokument se NE indeksira u bazu znanja — koristi se samo za analizu.",
            )

            if interni_fajl:
                st.success(f"✅ Odabran: **{interni_fajl.name}**")

            naziv_internog = st.text_input(
                "Naziv internog dokumenta",
                value=interni_fajl.name if interni_fajl else "",
                placeholder="npr. Pravilnik o internoj reviziji 2025",
                key="naziv_internog",
            )

            izvor_internog = st.text_input(
                "Izvor internog dokumenta",
                placeholder="npr. Interna izrada, Sektor za reviziju",
                key="izvor_internog",
            )

            godina_internog = st.text_input(
                "Godina internog dokumenta",
                placeholder="npr. 2025",
                key="godina_internog",
            )

        # ── Desna kolona: Odabir referentnih dokumenata ────────────────────────
        with col_r:
            st.subheader("📚 Referentni dokumenti (iz baze znanja)")
            st.caption("Odaberi jedan ili više dokumenata prema kojima se provjerava usklađenost.")

            # Grupiši po kategorijama za preglednost
            opcije_po_kat: dict[str, list[str]] = {}
            for dok in dokumenti_lista_compliance:
                kat = dok["kategorija"]
                if kat not in opcije_po_kat:
                    opcije_po_kat[kat] = []
                opcije_po_kat[kat].append(dok["naziv_dokumenta"])

            odabrani_referentni: list[str] = []

            for kat, nazivi in opcije_po_kat.items():
                st.markdown(f"**{kat}**")
                for naziv in nazivi:
                    if st.checkbox(naziv, key=f"ref_{naziv}"):
                        odabrani_referentni.append(naziv)

            if odabrani_referentni:
                st.info(f"✅ Odabrano: **{len(odabrani_referentni)}** referentnih dokumenata")
            else:
                st.warning("⚠️ Odaberi barem jedan referentni dokument.")

        st.divider()

        # ── Opcije analize ─────────────────────────────────────────────────────
        with st.expander("⚙️ Opcije analize", expanded=False):
            st.caption(
                "Napredne opcije za podešavanje analize. Defaultne vrijednosti su optimalne za većinu slučajeva."
            )
            dubina_analize = st.select_slider(
                "Dubina analize",
                options=["Površna", "Standardna", "Detaljna"],
                value="Standardna",
                help="Detaljna analiza traje duže ali daje preciznije rezultate.",
            )
            jezik_izvjestaja = st.radio(
                "Jezik izvještaja",
                options=["Srpski (ijekavica)", "Bosanski", "Engleski"],
                horizontal=True,
            )

        # ── Pokretanje analize ─────────────────────────────────────────────────
        analiza_moguca = bool(
            interni_fajl and odabrani_referentni and naziv_internog.strip()
        )

        if not analiza_moguca:
            st.info(
                "ℹ️ Za pokretanje analize potrebno je:\n"
                "1. Uploadovati interni dokument\n"
                "2. Unijeti naziv dokumenta\n"
                "3. Odabrati barem jedan referentni dokument"
            )

        if st.button(
            "⚖️ Pokreni analizu usklađenosti",
            disabled=not analiza_moguca,
            type="primary",
            use_container_width=True,
            key="btn_compliance",
        ):
            # Sačuvaj interni fajl privremeno
            sufiks = Path(interni_fajl.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=sufiks) as tmp:
                tmp.write(interni_fajl.read())
                tmp_putanja = Path(tmp.name)

            metadata_internog = {
                "naziv_dokumenta": naziv_internog.strip() or interni_fajl.name,
                "kategorija":      "Interni",
                "izvor":           izvor_internog.strip() or "Interna izrada",
                "godina":          godina_internog.strip() or str(datetime.now().year),
                "tip_dokumenta":   "Interni dokument za analizu",
            }

            st.subheader("📊 Izvještaj o usklađenosti")

            # Prikaz parametara analize
            with st.container():
                col_i, col_r2 = st.columns(2)
                with col_i:
                    st.markdown(
                        f"**Analizirani dokument:**  \n"
                        f"`{metadata_internog['naziv_dokumenta']}`"
                    )
                with col_r2:
                    st.markdown(
                        f"**Referentni dokumenti ({len(odabrani_referentni)}):**  \n"
                        + "  \n".join(f"• `{n}`" for n in odabrani_referentni)
                    )

            st.divider()

            # Streaming generacija izvještaja
            with st.spinner("🔍 Analiza u toku — ovo može trajati 30-90 sekundi..."):
                izvjestaj_placeholder = st.empty()
                kompletan_tekst = ""

                for token in analiziraj_uskladenost_stream(
                    kolekcija=kolekcija,
                    model=model,
                    putanja_dokumenta=tmp_putanja,
                    metadata_internog=metadata_internog,
                    nazivi_referentnih=odabrani_referentni,
                ):
                    kompletan_tekst += token
                    izvjestaj_placeholder.markdown(kompletan_tekst + "▌")

                # Finalni prikaz bez kursora
                izvjestaj_placeholder.markdown(kompletan_tekst)

            tmp_putanja.unlink(missing_ok=True)

            # Opcija za preuzimanje izvještaja
            st.divider()
            naziv_fajla = (
                f"uskladenost_{naziv_internog.replace(' ', '_')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            )
            st.download_button(
                label="⬇️ Preuzmi izvještaj (Markdown)",
                data=kompletan_tekst,
                file_name=naziv_fajla,
                mime="text/markdown",
                help="Izvještaj se sprema kao Markdown fajl koji možeš otvoriti u Word-u ili Notion-u.",
            )
