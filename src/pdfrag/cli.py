"""Command-line entry point: build the index (once) and run an interactive chat.

Run from anywhere after ``pip install -e .``:

    pdfrag                  # index only if empty, then chat
    pdfrag --reindex        # clear & rebuild the index, then chat
    python -m pdfrag        # same as `pdfrag`
"""

import sys

from pdfrag.config import DEFAULT_PDF, GROQ_API_KEY, GROQ_MODEL
from pdfrag.pipeline import answer_question, build_index
from pdfrag.retrieval import count_documents, reset_collection


def ensure_index(force=False):
    """Build the index only when needed, so re-runs don't duplicate vectors."""
    existing = count_documents()
    if force and existing:
        print(f"Reindex requested — clearing {existing} existing vectors...")
        reset_collection()
        existing = 0

    if existing == 0:
        print(f"Indexing {DEFAULT_PDF} ...")
        n = build_index(DEFAULT_PDF)
        print(f"Indexing complete — {n} chunks stored.")
    else:
        print(f"Using existing index ({existing} vectors). Pass --reindex to rebuild.")


def chat():
    """Interactive question/answer loop over the indexed document."""
    print("\nAsk questions about the document. Type 'q' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"q", "quit", "exit"}:
            break
        print("\n" + answer_question(question))


def main():
    if GROQ_API_KEY:
        print(f"Using Groq model: {GROQ_MODEL}")
    else:
        print(
            "⚠️  GROQ_API_KEY is not set in .env — answering will fail.\n"
            "   Get a free key at https://console.groq.com/keys\n"
        )
    ensure_index(force="--reindex" in sys.argv)
    chat()


if __name__ == "__main__":
    main()
