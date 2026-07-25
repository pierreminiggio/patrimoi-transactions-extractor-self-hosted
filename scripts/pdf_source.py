"""PDF text extraction. Unlike a hosted multimodal model, the local LLM has
no native vision, so we pull the text layer out of the PDF ourselves before
handing it to the model.
"""

import pdfplumber


def extract_pdf_text(path: str) -> str:
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # layout=True preserves the original spatial positions of words
            # (via whitespace padding), so table columns stay visually
            # separated instead of being collapsed into one run-on string.
            # x_tolerance controls how close two characters need to be
            # before pdfplumber treats them as part of the same "word" —
            # lowering it makes word-boundary detection stricter, which
            # matters for tightly-kerned table text.
            text = page.extract_text(layout=True, x_tolerance=1) or ""
            pages_text.append(text)
    full_text = "\n".join(pages_text)
    # NOTE: do not collapse internal whitespace runs. The horizontal position
    # of a number relative to column headers is often the only signal for
    # which column it belongs to (debit vs credit). Collapsing space runs
    # down to a fixed width destroys that signal. Trailing whitespace per
    # line is harmless to trim since it carries no column information.
    full_text = "\n".join(line.rstrip() for line in full_text.split("\n"))
    return full_text
