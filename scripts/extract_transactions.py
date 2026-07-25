import os
import sys

import common
import csv_source
import file_type
import llm_extract
import pdf_source

MODEL_PATH = os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
N_CTX = int(os.environ.get("LLM_N_CTX", "32768"))
CONTENT_TYPE = os.environ.get("SOURCE_CONTENT_TYPE")  # captured at download time, optional

if len(sys.argv) < 3:
    print("Usage: python extract_transactions.py <input_path> <output_json_path>")
    raise SystemExit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

_size = os.path.getsize(INPUT_PATH)
if _size == 0:
    print(
        f"ERROR: Downloaded file '{INPUT_PATH}' is 0 bytes — the statement "
        "itself was never downloaded. This is not a parsing problem; the "
        "download step returned an empty body. Likely causes: an expired or "
        "missing auth token (PATRIMOI_TOKEN), a wrong/expired statement URL, "
        "or the source endpoint returning 200 with an empty response for this "
        "particular URL. Check the 'Download bank statement' step's logs "
        "(Content-Type printed there) before re-running."
    )
    raise SystemExit(1)
print(f"Input file: {INPUT_PATH} ({_size} bytes), Content-Type hint: {CONTENT_TYPE or '<none>'}")


def run_via_llm(source_text: str):
    output_text = llm_extract.run_llm(source_text, MODEL_PATH, N_CTX)
    return llm_extract.parse_llm_output(output_text, source_text)


def handle_pdf(path: str):
    statement_text = pdf_source.extract_pdf_text(path)
    if not statement_text.strip():
        print(
            "ERROR: Could not extract any text from the PDF. It may be a scanned "
            "image rather than a text-based PDF, which this pipeline does not "
            "currently handle (no OCR step is included)."
        )
        raise SystemExit(1)
    return run_via_llm(statement_text)


def handle_csv(path: str):
    fieldnames, row_count = csv_source.inspect(path)
    print(f"CSV header detected: {fieldnames}")
    print(f"CSV data rows detected: {row_count}")
    if not fieldnames or row_count == 0:
        print(
            "ERROR: The downloaded file decodes as text but doesn't look like "
            "a CSV with a header row and at least one data row. If this is "
            "unexpected, check the 'Download bank statement' step's logs — "
            "this usually means the download returned something other than "
            "the actual statement (an error page, a redirect target, etc.) "
            "rather than a genuine CSV-parsing failure."
        )
        raise SystemExit(1)

    # Tier 1: known exact provider format (e.g. N26) — pure Python, no LLM.
    result = csv_source.try_known_format(path)
    if result is not None:
        parsed_rows, starting_balance, stated_ending_balance, format_name = result
        print(f"Recognized CSV format: {format_name}. Parsed deterministically — the model was never invoked.")
        return parsed_rows, starting_balance, stated_ending_balance

    # Tier 2: unrecognized provider, but columns can be guessed generically
    # and the guess validates against a sample of the actual data — still
    # pure Python, no LLM.
    result = csv_source.try_generic_parse(path)
    if result is not None:
        parsed_rows, starting_balance, stated_ending_balance, column_desc = result
        print(
            "CSV format not recognized as a known provider, but columns were "
            f"inferred generically ({column_desc}) and validated against the "
            "data. Parsed deterministically — the model was never invoked."
        )
        return parsed_rows, starting_balance, stated_ending_balance

    # Tier 3: fall back to the LLM, same treatment as PDF text.
    print(
        "Header and rows were found, but neither a known provider format nor "
        "a generic column guess could be confidently matched/validated "
        f"against the {row_count} data row(s) under header {fieldnames}. "
        "Falling back to the local LLM."
    )
    statement_text = csv_source.read_text(path)
    if not statement_text.strip():
        print("ERROR: The CSV file appears to be empty.")
        raise SystemExit(1)
    return run_via_llm(statement_text)


detected_type = file_type.detect_file_type(INPUT_PATH, CONTENT_TYPE)
print(f"Detected input type: {detected_type}")

if detected_type == "pdf":
    parsed_rows, starting_balance, stated_ending_balance = handle_pdf(INPUT_PATH)
else:
    parsed_rows, starting_balance, stated_ending_balance = handle_csv(INPUT_PATH)

common.finalize(parsed_rows, starting_balance, stated_ending_balance, OUTPUT_PATH)
