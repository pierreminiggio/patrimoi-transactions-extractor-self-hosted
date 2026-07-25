"""Detect whether the downloaded input file is a PDF or a CSV.

Strategy: check the HTTP Content-Type header captured at download time
first; if it's missing or ambiguous (e.g. application/octet-stream, which
many signed-URL/storage links send regardless of actual content), fall back
to sniffing the file's own bytes. If the header and the file's actual bytes
disagree, the bytes win — a wrong/generic header shouldn't be trusted over
directly-observable content.
"""

import csv


def _sniff_is_pdf(head: bytes) -> bool:
    return head.lstrip(b"\xef\xbb\xbf").startswith(b"%PDF-")


def _sniff_is_csv(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            raw = f.read(8192)
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return False
    except OSError:
        return False

    sample = text[:4096]
    if not sample.strip():
        return False
    try:
        csv.Sniffer().sniff(sample, delimiters=",;\t")
        return True
    except csv.Error:
        # Fall back to a crude check: does the first line have multiple
        # consistent delimiter-separated fields?
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        for delim in (",", ";", "\t"):
            if first_line.count(delim) >= 1:
                return True
        return False


def _content_type_hint(content_type: str):
    if not content_type:
        return None
    ct = content_type.lower()
    if "pdf" in ct:
        return "pdf"
    if "csv" in ct or "text/plain" in ct or "comma-separated" in ct:
        return "csv"
    return None


def detect_file_type(path: str, content_type: str = None) -> str:
    with open(path, "rb") as f:
        head = f.read(2048)

    is_pdf_by_magic = _sniff_is_pdf(head)
    hint = _content_type_hint(content_type)

    if is_pdf_by_magic:
        if hint == "csv":
            print(
                "WARNING: Content-Type header said CSV but the file starts with "
                "a PDF signature (%PDF-); trusting the file content and "
                "treating this as a PDF."
            )
        return "pdf"

    if hint == "pdf":
        print(
            "WARNING: Content-Type header said PDF but the file content does "
            "not start with a PDF signature; falling back to content sniffing."
        )
    elif hint == "csv":
        return "csv"

    if _sniff_is_csv(path):
        return "csv"

    print(
        "ERROR: Could not determine whether the downloaded file is a PDF or "
        "a CSV (Content-Type was missing/ambiguous, and the content doesn't "
        "look like either a PDF or a well-formed CSV)."
    )
    raise SystemExit(1)
