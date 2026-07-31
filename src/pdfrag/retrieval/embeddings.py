"""Local, free embeddings via HuggingFace sentence-transformers."""

from langchain_huggingface import HuggingFaceEmbeddings

from pdfrag.config import EMBEDDING_MODEL

_embeddings = None


def get_embeddings():
    """Return a lazily-initialised, cached embeddings object.

    This is a LangChain ``Embeddings`` instance, so it plugs directly into the
    Chroma vector store — we never embed text by hand.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings
