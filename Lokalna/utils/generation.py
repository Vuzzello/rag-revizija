# Verzija: 1.0 | Ažurirano: 2025-04-27

import logging
from typing import Generator

import ollama

from config import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Ti si ekspert za internu reviziju. Odgovaraj isključivo na osnovu priloženog konteksta.
Ako odgovor nije u kontekstu, jasno kaži: 'Nisam pronašao relevantan odgovor u bazi znanja.'
Nikada ne izmišljaj činjenice. Budi precizan, profesionalan i koristi revizorsku terminologiju.
Na kraju odgovora uvijek navedi izvore koje si koristio u formatu:
Korišteni izvori:
- [naziv dokumenta] | [izvor] | [godina]"""


def provjeri_ollama() -> bool:
    """Provjerava da li je Ollama servis dostupan i model učitan."""
    try:
        klijent = ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)
        modeli  = klijent.list()
        nazivi  = [m.model for m in modeli.models]
        dostupan = any(OLLAMA_MODEL in naziv for naziv in nazivi)
        if not dostupan:
            logger.warning(
                f"Model '{OLLAMA_MODEL}' nije pronađen. "
                f"Dostupni modeli: {nazivi}"
            )
        return dostupan
    except Exception as e:
        logger.error(f"Ollama nije dostupan na {OLLAMA_URL}: {e}")
        return False


def kreiraj_prompt(upit: str, kontekst: str) -> str:
    """Kreira finalni prompt sa upitom i kontekstom za LLM."""
    return (
        f"Na osnovu sljedećeg konteksta odgovori na pitanje.\n\n"
        f"=== KONTEKST ===\n{kontekst}\n\n"
        f"=== PITANJE ===\n{upit}"
    )


def generiraj_odgovor_stream(
    upit: str,
    kontekst: str,
) -> Generator[str, None, None]:
    """
    Generira odgovor putem Ollama streaming API-ja.
    Yield-a tekst token po token za Streamlit st.write_stream().
    """
    if not kontekst.strip():
        yield "Nisam pronašao relevantan odgovor u bazi znanja."
        return

    prompt = kreiraj_prompt(upit, kontekst)

    try:
        klijent = ollama.Client(host=OLLAMA_URL, timeout=OLLAMA_TIMEOUT)

        stream = klijent.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            stream=True,
            #options={
                #"temperature": OLLAMA_TEMPERATURE,
                #"num_predict": 1024,
            options={
                "temperature": 0.1,           # Smanjujemo kreativnost na minimum
                "num_predict": 1024,
                "repeat_penalty": 1.5,        # KAŽNJAVA ponavljanje istih rečenica
                "repeat_last_n": 128,         # Koliko daleko unazad model gleda da ne ponovi tekst
                "top_k": 20,                  # Ograničava izbor riječi na najvjerovatnije
                "top_p": 0.5                  # Dodatno filtriranje fokusa
            },
        )

        for dio in stream:
            token = dio.message.content
            if token:
                yield token

    #except ollama.ResponseError as e:
        #logger.error(f"Ollama greška pri generaciji: {e}")
        #yield f"Greška pri generaciji odgovora: {e}"
    #except Exception as e:
        #logger.error(f"Neočekivana greška pri generaciji: {e}")
        #yield "Desila se greška. Provjeri da li Ollama radi (`ollama serve`)."

    except ollama.ResponseError as e:
        logger.error(f"Ollama greška pri generaciji: {e}")
        yield f"⚠️ **Ollama ResponseError:** {e.error}"
    except Exception as e:
        logger.error(f"Neočekivana greška pri generaciji: {e}")
        # Ovo će ispisati tačan naziv greške i opis u Streamlit chat-u
        yield f"❌ **STVARNA GREŠKA:** {type(e).__name__}: {str(e)}"

def generiraj_odgovor(upit: str, kontekst: str) -> str:
    """
    Generira kompletan odgovor (bez streaminga).
    Koristi se za testiranje i debugging.
    """
    return "".join(generiraj_odgovor_stream(upit, kontekst))
