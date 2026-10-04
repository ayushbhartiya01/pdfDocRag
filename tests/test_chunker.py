from langchain_core.documents import Document

from pdfrag.ingestion.chunker import create_chunks


def test_create_chunks_splits_text_and_preserves_metadata():
    source = Document(
        page_content="abcdefghij",
        metadata={"source": "paper.pdf", "page": 2},
    )

    chunks = create_chunks([source], chunk_size=5, chunk_overlap=2)

    assert [chunk.page_content for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert all(chunk.metadata == source.metadata for chunk in chunks)


def test_create_chunks_returns_empty_list_for_no_documents():
    assert create_chunks([]) == []