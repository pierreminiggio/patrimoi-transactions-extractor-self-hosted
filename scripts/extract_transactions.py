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
        "Could not confidently parse this CSV deterministically (unrecognized "
        "provider, and generic column guesses didn't validate against the "
        "data). Falling back to the local LLM."
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
