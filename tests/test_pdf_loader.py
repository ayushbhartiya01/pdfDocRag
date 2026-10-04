from pdfrag.ingestion.pdf_loader import _columnise


def word(text, x0, x1, top):
    return {"text": text, "x0": x0, "x1": x1, "top": top}


def test_columnise_reads_left_column_before_right_column():
    words = [
        word("right one", 350, 420, 100),
        word("left one", 50, 120, 100),
        word("right two", 350, 420, 120),
        word("left two", 50, 120, 120),
    ]

    assert _columnise(words, page_width=600) == (
        "left one\nleft two\nright one\nright two"
    )


def test_columnise_keeps_full_width_heading_before_columns():
    words = [
        word("Heading", 50, 110, 50),
        word("for paper", 115, 190, 50),
        word("right text", 350, 420, 100),
        word("left text", 50, 120, 100),
    ]

    assert _columnise(words, page_width=600) == (
        "Heading for paper\nleft text\nright text"
    )


def test_columnise_handles_empty_page():
    assert _columnise([], page_width=600) == ""