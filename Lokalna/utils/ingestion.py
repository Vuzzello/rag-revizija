# Verzija: 1.0 | Ažurirano: 2025-04-27

import logging
from pathlib import Path

import pypdf
import docx

from config import PODRZANI_FORMATI

logger = logging.getLogger(__name__)


def parsiraj_pdf(putanja: Path) -> str:
    """Čita tekst iz PDF fajla stranica po stranica."""
    tekst_stranica = []
    try:
        citac = pypdf.PdfReader(str(putanja))
        for i, stranica in enumerate(citac.pages):
            sadrzaj = stranica.extract_text()
            if sadrzaj:
                tekst_stranica.append(sadrzaj)
            else:
                logger.warning(f"Stranica {i + 1} u '{putanja.name}' je prazna ili nečitljiva.")
        return "\n\n".join(tekst_stranica)
    except Exception as e:
        logger.error(f"Greška pri parsiranju PDF-a '{putanja.name}': {e}")
        return ""


def parsiraj_docx(putanja: Path) -> str:
    """Čita tekst iz DOCX fajla paragraf po paragraf."""
    try:
        dokument = docx.Document(str(putanja))
        paragrafi = [p.text for p in dokument.paragraphs if p.text.strip()]
        return "\n\n".join(paragrafi)
    except Exception as e:
        logger.error(f"Greška pri parsiranju DOCX-a '{putanja.name}': {e}")
        return ""


def parsiraj_txt(putanja: Path) -> str:
    """Čita tekst iz TXT fajla (UTF-8 enkodiranje)."""
    try:
        return putanja.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Greška pri parsiranju TXT-a '{putanja.name}': {e}")
        return ""


def ucitaj_dokument(putanja: Path, metadata: dict) -> dict | None:
    """
    Parsira dokument i vraća dict sa tekstom i metadatom.
    Vraća None ako fajl nije podržan ili ako je tekst prazan.
    """
    sufiks = putanja.suffix.lower()

    if sufiks not in PODRZANI_FORMATI:
        logger.warning(f"Format '{sufiks}' nije podržan: '{putanja.name}'")
        return None

    if sufiks == ".pdf":
        tekst = parsiraj_pdf(putanja)
    elif sufiks == ".docx":
        tekst = parsiraj_docx(putanja)
    elif sufiks == ".txt":
        tekst = parsiraj_txt(putanja)

    tekst = tekst.strip()

    if not tekst:
        logger.warning(f"Dokument '{putanja.name}' je prazan nakon parsiranja — preskačem.")
        return None

    logger.info(f"Uspješno učitan dokument '{putanja.name}' ({len(tekst)} znakova).")

    return {
        "tekst": tekst,
        "metadata": {
            **metadata,
            "naziv_dokumenta": putanja.name,
        },
    }


def ucitaj_vise_dokumenata(putanje: list[Path], metadata: dict) -> list[dict]:
    """Parsira listu fajlova i vraća samo uspješno učitane dokumente."""
    rezultati = []
    for putanja in putanje:
        dokument = ucitaj_dokument(putanja, metadata)
        if dokument:
            rezultati.append(dokument)
    logger.info(f"Učitano {len(rezultati)} od {len(putanje)} dokumenata.")
    return rezultati
