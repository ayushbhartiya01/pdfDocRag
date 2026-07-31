"""Embed and persist chunks into the Chroma vector store."""

from pdfrag.retrieval.vector_store import get_vector_store


def store_chunks(chunks):
    """Add chunk Documents to the vector store.

    Chroma embeds them via the store's embedding function and persists to disk
    automatically (no manual ids/embeddings needed).

    Args:
        chunks (list[Document]): Chunked Documents to store.
    """
    if not chunks:
        return
    get_vector_store().add_documents(chunks)
