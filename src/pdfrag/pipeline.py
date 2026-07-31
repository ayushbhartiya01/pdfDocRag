"""The guarded RAG pipeline — ties retrieval, generation and guardrails together.

The query flow is expressed as composed LCEL steps so the guardrails are
genuinely *in* the pipeline:

    input-guard -> retrieve -> grounding-guard -> generate -> output-guard

Each step receives and returns a small ``state`` dict and short-circuits
(passes through untouched) once an answer or refusal has been set.
"""

from langchain_core.runnables import RunnableLambda

from pdfrag import guardrails
from pdfrag.generation import ask_llm
from pdfrag.ingestion import create_chunks, load_pdf
from pdfrag.retrieval import search, store_chunks


# --- Indexing ------------------------------------------------------------


def build_index(pdf_path):
    """Load a PDF, chunk it, and store the chunks in the vector store.

    Returns:
        int: Number of chunks stored.
    """
    documents = load_pdf(pdf_path)
    chunks = create_chunks(documents)
    store_chunks(chunks)
    return len(chunks)


# --- Query pipeline steps ------------------------------------------------


def _guard_input(state):
    ok, message = guardrails.check_input(state["query"])
    if not ok:
        state["answer"] = message
        state["blocked"] = True
    return state


def _retrieve(state):
    if state.get("blocked"):
        return state
    state["scored_docs"] = search(state["query"])
    return state


def _guard_grounding(state):
    if state.get("blocked"):
        return state
    if not guardrails.check_grounding(state["scored_docs"]):
        state["answer"] = guardrails.REFUSAL_NO_CONTEXT
        state["blocked"] = True
    return state


def _format_context(scored_docs):
    """Join retrieved chunks into a single context string, tagging pages."""
    parts = []
    for doc, _score in scored_docs:
        page = doc.metadata.get("page")
        tag = f"[page {page + 1}] " if isinstance(page, int) else ""
        parts.append(tag + doc.page_content)
    return "\n\n".join(parts)


def _generate(state):
    if state.get("blocked"):
        return state
    context = _format_context(state["scored_docs"])
    state["answer"] = ask_llm(context=context, question=state["query"])
    return state


def _guard_output(state):
    # Final safety pass — strip any PII the model may have surfaced.
    state["answer"] = guardrails.redact_pii(state["answer"])
    return state


# Compose the steps into a single LCEL Runnable.
pipeline = (
    RunnableLambda(_guard_input)
    | RunnableLambda(_retrieve)
    | RunnableLambda(_guard_grounding)
    | RunnableLambda(_generate)
    | RunnableLambda(_guard_output)
)


def answer_question(query):
    """Run the full guarded RAG pipeline and return the answer string."""
    state = pipeline.invoke({"query": query, "blocked": False})
    return state["answer"]
