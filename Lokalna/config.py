# Verzija: 1.0 | Ažurirano: 2025-04-27

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Učitaj .env fajl
load_dotenv()

# ── Logging konfiguracija ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── LLM ───────────────────────────────────────────────────────────────────────
#OLLAMA_MODEL: str         = os.getenv("OLLAMA_MODEL", "qwen3:latest")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b") #str    = os.getenv("OLLAMA_MODEL", "qwen3:latest")
OLLAMA_URL: str           = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT: int       = 300
OLLAMA_TEMPERATURE: float = 0.1

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")

# ── Chroma ─────────────────────────────────────────────────────────────────────
CHROMA_PATH: Path      = Path(os.getenv("CHROMA_PATH", "./db"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "revizija_kb")

# ── Podaci ─────────────────────────────────────────────────────────────────────
DATA_PATH: Path = Path(os.getenv("DATA_PATH", "./data/raw"))

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int         = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP: int      = int(os.getenv("CHUNK_OVERLAP", 200))
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int             = int(os.getenv("RETRIEVAL_TOP_K", 5))
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", 0.35))

# ── Kategorije dokumenata ──────────────────────────────────────────────────────
KATEGORIJE: list[str] = [
    "Regulatorni",
    "Operativni",
    "Ekspertize",
    "Opšti",
    "Ostali",
]

# ── Podržani formati fajlova ───────────────────────────────────────────────────
PODRZANI_FORMATI: list[str] = [".pdf", ".docx", ".txt"]

# ── Kreiraj potrebne direktorijume ako ne postoje ─────────────────────────────
CHROMA_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)
