"""Embeddings, vector store, indexing and search."""

from pdfrag.retrieval.embeddings import get_embeddings
from pdfrag.retrieval.indexer import store_chunks
from pdfrag.retrieval.search import get_retriever, search
from pdfrag.retrieval.vector_store import (
    count_documents,
    get_vector_store,
    reset_collection,
)

__all__ = [
    "get_embeddings",
    "get_vector_store",
    "count_documents",
    "reset_collection",
    "store_chunks",
    "search",
    "get_retriever",
]
