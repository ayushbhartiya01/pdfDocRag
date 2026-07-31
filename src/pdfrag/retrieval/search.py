"""Similarity search over the vector store, with relevance scores."""

from pdfrag.config import TOP_K
from pdfrag.retrieval.vector_store import get_vector_store


def search(query, top_k=TOP_K):
    """Return the most relevant chunks for a query, with scores.

    Args:
        query (str): The search query.
        top_k (int): Number of results to return.

    Returns:
        list[tuple[Document, float]]: ``(document, relevance_score)`` pairs.
        Scores are cosine-based (~[0, 1], higher = more relevant) so the
        grounding guardrail can judge whether the context is actually relevant.
    """
    return get_vector_store().similarity_search_with_relevance_scores(
        query, k=top_k
    )


def get_retriever(top_k=TOP_K):
    """Return a LangChain retriever for use inside LCEL chains."""
    return get_vector_store().as_retriever(search_kwargs={"k": top_k})
