"""Persistent Chroma vector store wired to the local embeddings."""

from langchain_chroma import Chroma

from pdfrag.config import CHROMA_DIR, COLLECTION_NAME
from pdfrag.retrieval.embeddings import get_embeddings

_vector_store = None


def get_vector_store():
    """Return a lazily-initialised, cached Chroma vector store.

    Configured for cosine space so the relevance scores returned by search are
    interpretable (~[0, 1]) for the grounding guardrail.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def count_documents():
    """Number of vectors currently stored (used to decide whether to index)."""
    return len(get_vector_store().get(include=[])["ids"])


def reset_collection():
    """Drop all stored vectors so the index can be rebuilt without duplicates."""
    global _vector_store
    get_vector_store().delete_collection()
    _vector_store = None  # force a fresh collection on next access
