"""Split loaded Documents into overlapping chunks for retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdfrag.config import CHUNK_OVERLAP, CHUNK_SIZE


def create_chunks(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split Documents into smaller chunks, preserving their metadata.

    Args:
        documents (list[Document]): Documents to split (e.g. PDF pages).
        chunk_size (int): Target maximum characters per chunk.
        chunk_overlap (int): Characters shared between adjacent chunks.

    Returns:
        list[Document]: The chunked Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)
