"""Detect whether the downloaded input file is a PDF or a CSV.

Strategy: check the HTTP Content-Type header captured at download time
first; if it's missing or ambiguous (e.g. application/octet-stream, which
many signed-URL/storage links send regardless of actual content), fall back
to sniffing the file's own bytes. If the header and the file's actual bytes
disagree, the bytes win — a wrong/generic header shouldn't be trusted over
directly-observable content.

Since this pipeline only ever expects a PDF or a CSV, detection is a binary
choice: PDF magic bytes (%PDF-) mean PDF; anything else that decodes as text
is treated as CSV and handed to the CSV parsing tiers, which are better
placed than this module to judge whether the content actually makes sense
as a statement (and will say so clearly if not). Only genuinely undecodable
binary content — which usually means the download itself went wrong (auth
failure, wrong URL, HTML error page served as bytes, etc.) — is a hard
failure here, and it prints diagnostics to make that easy to spot in logs.
"""


def _sniff_is_pdf(head: bytes) -> bool:
    return head.lstrip(b"\xef\xbb\xbf").startswith(b"%PDF-")


def _decode_text(raw: bytes):
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


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
        head = f.read(8192)

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

    text = _decode_text(head)
    if text is not None:
        return "csv"

    import os

    print(
        "ERROR: Downloaded file is neither a PDF (no %PDF- signature) nor "
        "decodable as text, so it can't be a CSV either. This usually means "
        "the download itself failed silently — e.g. wrong/expired auth token, "
        "wrong URL, or the server returned an HTML/JSON error page instead of "
        "the actual statement (curl -f would normally catch a non-2xx status, "
        "but not a 2xx response with unexpected content)."
    )
    print(f"  Content-Type header seen: {content_type or '<none>'}")
    print(f"  Downloaded file size: {os.path.getsize(path)} bytes")
    print(f"  First bytes (hex): {head[:64].hex()}")
    raise SystemExit(1)
