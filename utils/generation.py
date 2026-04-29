# Verzija: 2.0 | Ažurirano: 2025-04-27
# Generacija odgovora putem Groq API (Llama 3.3 70B)

import logging
from typing import Generator

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ti si asistent za interne revizore. Odgovaraj isključivo na osnovu priloženog konteksta.
Ako odgovor nije u kontekstu, jasno kaži: 'Nisam pronašao relevantan odgovor u bazi znanja.'
Nikada ne izmišljaj činjenice. Budi precizan i koncizan.
Na kraju odgovora uvijek navedi izvore koje si koristio u formatu:
Korišteni izvori:
- [naziv dokumenta] | [izvor] | [godina]"""

# ── Compliance system prompt ───────────────────────────────────────────────────
COMPLIANCE_SYSTEM_PROMPT = """Ti si ekspert za pravnu i regulatornu usklađenost dokumenata u oblasti interne revizije.
Tvoj zadatak je da analiziraš interni dokument i pronađeš mjere u kojima nije usklađen sa referentnim dokumentima.

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

Budi precizan i navodi konkretne odredbe. Ne izmišljaj reference koje nisu u kontekstu."""


def _kreiraj_groq_klijent() -> Groq:
    """Kreira Groq klijent."""
    return Groq(api_key=GROQ_API_KEY)


def provjeri_groq() -> bool:
    """Provjerava dostupnost Groq API-ja."""
    try:
        klijent = _kreiraj_groq_klijent()
        klijent.models.list()
        return True
    except Exception as e:
        logger.error(f"Groq API nije dostupan: {e}")
        return False


def generiraj_odgovor_stream(
    upit: str,
    kontekst: str,
) -> Generator[str, None, None]:
    """
    Generira odgovor putem Groq streaming API-ja.
    Yield-a tekst token po token za Streamlit st.write_stream().
    """
    if not kontekst.strip():
        yield "Nisam pronašao relevantan odgovor u bazi znanja."
        return

    prompt = (
        f"Na osnovu sljedećeg konteksta odgovori na pitanje.\n\n"
        f"=== KONTEKST ===\n{kontekst}\n\n"
        f"=== PITANJE ===\n{upit}"
    )

    try:
        klijent = _kreiraj_groq_klijent()
        stream  = klijent.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except Exception as e:
        logger.error(f"Groq greška pri generaciji: {e}")
        yield f"\n\n❌ Greška pri generaciji odgovora: {e}"


def generiraj_compliance_stream(
    interni_tekst: str,
    referentni_kontekst: str,
    naziv_internog: str,
) -> Generator[str, None, None]:
    """
    Generira compliance izvještaj putem Groq streaming API-ja.
    """
    if not referentni_kontekst.strip():
        yield "⚠️ Nisu pronađeni relevantni odlomci u referentnim dokumentima."
        return

    prompt = (
        f"=== INTERNI DOKUMENT ZA ANALIZU ===\n"
        f"Naziv: {naziv_internog}\n\n"
        f"{interni_tekst[:4000]}\n\n"
        f"=== REFERENTNI DOKUMENTI ===\n"
        f"{referentni_kontekst}"
        f"\n\nAnaliziraj usklađenost internog dokumenta sa referentnim dokumentima."
    )

    try:
        klijent = _kreiraj_groq_klijent()
        stream  = klijent.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.05,
            max_tokens=3000,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    except Exception as e:
        logger.error(f"Groq greška pri compliance analizi: {e}")
        yield f"\n\n❌ Greška: {e}"


def generiraj_odgovor(upit: str, kontekst: str) -> str:
    """Vraća kompletan odgovor (bez streaminga) — za testiranje."""
    return "".join(generiraj_odgovor_stream(upit, kontekst))
