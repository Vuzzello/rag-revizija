# Verzija: 2.0 | Ažurirano: 2025-04-27

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Supabase ───────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# ── Groq LLM ───────────────────────────────────────────────────────────────────
GROQ_API_KEY: str     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE: float = 0.1
GROQ_MAX_TOKENS: int  = 2048

# ── HuggingFace Embeddings ─────────────────────────────────────────────────────
HF_API_TOKEN: str    = os.getenv("HF_API_TOKEN", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
HF_API_URL: str      = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int         = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int      = int(os.getenv("CHUNK_OVERLAP", 200))
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int             = int(os.getenv("RETRIEVAL_TOP_K", 5))
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.35))

# ── Kategorije ─────────────────────────────────────────────────────────────────
KATEGORIJE: list[str] = [
    "Regulatorni",
    "Operativni",
    "Ekspertize",
    "Opšti",
    "Ostali",
]

# ── Podržani formati ───────────────────────────────────────────────────────────
PODRZANI_FORMATI: list[str] = [".pdf", ".docx", ".txt"]

# ── Provjera obaveznih varijabli ──────────────────────────────────────────────
def provjeri_konfiguraciju() -> list[str]:
    """Vraća listu nedostajućih varijabli."""
    nedostaje = []
    if not SUPABASE_URL:
        nedostaje.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        nedostaje.append("SUPABASE_KEY")
    if not GROQ_API_KEY:
        nedostaje.append("GROQ_API_KEY")
    if not HF_API_TOKEN:
        nedostaje.append("HF_API_TOKEN")
    return nedostaje
