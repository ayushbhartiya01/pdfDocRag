"""Central configuration for the pdfrag pipeline.

All environment loading, filesystem paths, model names, and tunable knobs live
here so the rest of the code never has to touch ``os.getenv`` directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root from this file (src/pdfrag/config.py -> project root)
# so paths work regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# --- Paths ---------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PDF = str(DATA_DIR / "sample.pdf")
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = "pdf_embeddings"

# --- Models --------------------------------------------------------------
# Embeddings run locally and free via sentence-transformers.
EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# Generation runs on Groq's free tier (https://console.groq.com).
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# --- Chunking / retrieval ------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Guardrails ----------------------------------------------------------
MIN_QUERY_LEN = int(os.getenv("MIN_QUERY_LEN", "3"))
MAX_QUERY_LEN = int(os.getenv("MAX_QUERY_LEN", "1000"))
# Chroma is configured for cosine space, so relevance scores approximate
# cosine similarity in [0, 1]. If the best-matching chunk scores below this,
# we treat the question as unanswerable from the document (anti-hallucination).
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))
