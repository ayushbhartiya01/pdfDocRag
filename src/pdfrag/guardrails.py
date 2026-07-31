"""Custom, dependency-free guardrails for the RAG pipeline.

Three layers, all plain Python + regex (free, transparent, no external calls):

  1. Input guardrails    -- validate/sanitise the question before it reaches
     retrieval or the LLM (length checks + prompt-injection / jailbreak detection).
  2. Grounding guardrail -- refuse to answer when retrieval finds nothing
     relevant, which prevents the model from hallucinating.
  3. Output guardrails   -- redact PII from the answer before it is shown.
"""

import re

from pdfrag.config import MAX_QUERY_LEN, MIN_QUERY_LEN, RELEVANCE_THRESHOLD

# Safe, user-facing messages.
REFUSAL_NO_CONTEXT = "I don't have enough information in the document to answer that."
REFUSAL_INJECTION = (
    "Your request looks like an attempt to change my instructions, so I can't "
    "process it. Please ask a question about the document."
)

# --- 1. Input guardrails -------------------------------------------------

# Patterns that signal an attempt to override the system instructions or
# extract the prompt (prompt-injection / jailbreak).
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context)",
    r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"forget\s+(everything|all|the\s+above|(your|the)\s+(previous\s+)?instructions?)",
    r"(reveal|show|print|repeat|reprint|tell\s+me)\s+(your|the)\s+(system\s+)?(prompt|instructions?)",
    r"(your|the)\s+system\s+prompt",
    r"you\s+are\s+now\b",
    r"act\s+as\s+(if|a|an|though)\b",
    r"pretend\s+(to\s+be|you\s+are)\b",
    r"develop(er)?\s+mode",
    r"jailbreak",
    r"\bDAN\b",
    r"override\s+(your|the)\s+(instructions?|rules|guardrails?)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_input(query):
    """Validate a user query.

    Returns:
        tuple[bool, str]: ``(ok, message)``. When ``ok`` is False, ``message``
        is a safe explanation to show the user; when True, ``message`` is "".
    """
    if not query or not query.strip():
        return False, "Please enter a question."

    text = query.strip()
    if len(text) < MIN_QUERY_LEN:
        return False, "That question is too short. Please add more detail."
    if len(text) > MAX_QUERY_LEN:
        return False, f"That question is too long (max {MAX_QUERY_LEN} characters)."
    if _INJECTION_RE.search(text):
        return False, REFUSAL_INJECTION
    return True, ""


# --- 2. Grounding guardrail ---------------------------------------------


def check_grounding(scored_docs):
    """Decide whether the retrieved context is relevant enough to answer.

    Args:
        scored_docs (list[tuple[Document, float]]): Output of ``search`` —
            ``(document, relevance_score)`` pairs.

    Returns:
        bool: True if at least one chunk meets ``RELEVANCE_THRESHOLD``.
    """
    if not scored_docs:
        return False
    best_score = max(score for _, score in scored_docs)
    return best_score >= RELEVANCE_THRESHOLD


# --- 3. Output guardrails ------------------------------------------------

# Ordered most-specific first so patterns don't cannibalise each other.
_PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{1,4}\b"), "[REDACTED_CARD]"),
    (re.compile(r"(?<!\w)\+?\d[\d().\s-]{7,}\d(?!\w)"), "[REDACTED_PHONE]"),
]


def redact_pii(text):
    """Redact emails, SSNs, card-like numbers and phone numbers from text.

    Heuristic and safety-biased: it may occasionally over-redact a long plain
    number, which is the safer failure mode for a guardrail.
    """
    if not text:
        return text
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
