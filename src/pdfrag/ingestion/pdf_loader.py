"""Load PDFs into LangChain ``Document`` objects with column-aware extraction.

arXiv-style papers are typeset in two columns. A naive top-to-bottom extraction
interleaves the two columns into nonsense. This loader uses ``pdfplumber`` word
coordinates to read each column in order, while leaving full-width elements
(titles, tables, figure captions, single-column text) in place.

Per page:
  1. Group words into rows by vertical position.
  2. Segment each row: if it spans the page centre and has a wide central gap
     (the column gutter), split it into a left and a right segment; otherwise
     it is a full-width line, or a single-column (left- or right-only) line.
  3. Walk rows top-to-bottom, buffering left- and right-column segments and
     flushing them (all left, then all right) whenever a full-width line
     separates two-column blocks, and once at the end.

This degrades gracefully: a single-column page is all full-width lines, so it
comes out in plain reading order.

Worked example (real coordinates from data/sample.pdf, page 1; page width = 612
so center = 306, and _EDGE_MARGIN makes the centre band 301..311):

  Title row (top = 82.5):
    "A"(x0=79.9, x1=92.0)  "Systematic"(96.3, 180.8) ... "Reliable"(467.4, 531.5)
    row x0 = 79.9 (< 301) and x1 = 531.5 (> 311)          -> spans the centre
    widest gap between words = 4.3 pt  (< _GUTTER_MIN = 18) -> NOT a gutter
    => [("full", row)]    the title is kept whole and emitted in place

  Author row (tops 128.3 and 130.2 -> same row, within _LINE_TOL = 3):
    "SofiaBennani"(147.9, 218.4)   |   "CharlesMoslonka"(385.5, 473.7)
    row x0 = 147.9 (< 301) and x1 = 473.7 (> 311)         -> spans the centre
    widest gap = 385.5 - 218.4 = 167.1 pt (>= 18), gap_x = 301.95 (~ center)
    => [("left", "SofiaBennani"), ("right", "CharlesMoslonka")]  split at gutter

  No full-width line follows until the page ends, so the left/right buffers
  collect the whole of each column; the final flush emits the title, then the
  entire left column, then the entire right column -> correct reading order.
"""

import pdfplumber
from langchain_core.documents import Document

_LINE_TOL = 3.0      # points: vertical gap within which words share a row
_GUTTER_MIN = 18.0   # points: min horizontal gap to count as a column gutter
_EDGE_MARGIN = 5.0   # points: slack around the centre for "spans the centre"
_CENTRAL = 0.2       # gutter must sit within ±(this × width) of the page centre


def _rows(words):
    """Group words into rows by vertical position, ordered top-to-bottom."""
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    rows, current, top = [], [], None
    for w in words:
        if top is None or abs(w["top"] - top) <= _LINE_TOL:
            current.append(w)
            top = w["top"] if top is None else top
        else:
            rows.append(current)
            current, top = [w], w["top"]
    if current:
        rows.append(current)
    return rows


def _text(segment):
    return " ".join(w["text"] for w in sorted(segment, key=lambda w: w["x0"]))


def _segment_row(row, center, page_width):
    """Split a row into [(kind, words), ...] with kind in {full, left, right}."""
    ws = sorted(row, key=lambda w: w["x0"])
    x0 = ws[0]["x0"]
    x1 = max(w["x1"] for w in ws)
    spans_center = x0 < center - _EDGE_MARGIN and x1 > center + _EDGE_MARGIN

    best_gap, gap_x = 0.0, None
    for a, b in zip(ws, ws[1:]):
        gap = b["x0"] - a["x1"]
        if gap > best_gap:
            best_gap, gap_x = gap, (a["x1"] + b["x0"]) / 2

    # Two-column row: spans the centre AND has a wide gap near the centre.
    if (
        spans_center
        and best_gap >= _GUTTER_MIN
        and gap_x is not None
        and abs(gap_x - center) <= page_width * _CENTRAL
    ):
        left = [w for w in ws if w["x1"] <= gap_x]
        right = [w for w in ws if w["x0"] >= gap_x]
        if left and right:
            return [("left", left), ("right", right)]

    if spans_center:
        return [("full", ws)]  # continuous wide line: title / full-width table row
    return [("left" if (x0 + x1) / 2 < center else "right", ws)]  # single-column line


def _columnise(words, page_width):
    """Return page text in human reading order, de-columning two-column layouts."""
    if not words:
        return ""
    center = page_width / 2
    out, left_buf, right_buf = [], [], []

    def flush():
        out.extend(left_buf)
        out.extend(right_buf)
        left_buf.clear()
        right_buf.clear()

    for row in _rows(words):
        for kind, segment in _segment_row(row, center, page_width):
            if kind == "full":
                flush()
                out.append(_text(segment))
            elif kind == "left":
                left_buf.append(_text(segment))
            else:
                right_buf.append(_text(segment))
    flush()
    return "\n".join(out)


def load_pdf(file_path):
    """Load a PDF into a list of Documents (one per non-empty page).

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list[Document]: Page Documents with column-aware text and
        ``metadata['page']`` (0-indexed) + ``metadata['source']``.
    """
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            # Keep only upright words — drops rotated margin stamps (e.g. the
            # sideways "arXiv:… [cs.CL]" watermark) that would otherwise be noise.
            words = [w for w in page.extract_words(use_text_flow=False) if w.get("upright", True)]
            text = _columnise(words, page.width)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(file_path), "page": page_number},
                    )
                )
    return documents
