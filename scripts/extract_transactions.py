import json
import os
import sys

import pdfplumber
from llama_cpp import Llama

# --- Configuration via environment / args ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/qwen2.5-7b-instruct-q4_k_m.gguf")
N_CTX = int(os.environ.get("LLM_N_CTX", "32768"))

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}.")
    raise SystemExit(1)

if len(sys.argv) < 3:
    print("Usage: python extract_transactions.py <input_pdf_path> <output_json_path>")
    raise SystemExit(1)

PDF_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]


# --- Step 1: extract raw text from the PDF ---
# Unlike a hosted multimodal model, the local LLM has no native vision, so we
# pull the text layer out of the PDF ourselves before handing it to the model.
def extract_pdf_text(path: str) -> str:
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n".join(pages_text)


statement_text = extract_pdf_text(PDF_PATH)

if not statement_text.strip():
    print(
        "ERROR: Could not extract any text from the PDF. It may be a scanned "
        "image rather than a text-based PDF, which this self-hosted pipeline "
        "does not currently handle (no OCR step is included)."
    )
    raise SystemExit(1)

# --- Step 2: build the prompt ---
# Note the model is only asked to extract raw fields (date, description,
# debit, credit) plus the single overall starting balance. The running
# balance per transaction is computed deterministically in Python below,
# rather than trusted to the model's arithmetic, which is more reliable on
# a small local model than asking it to carry a running sum across dozens
# of rows.
prompt = f"""You are given the raw text extracted from a bank statement PDF. The text below may have imperfect spacing or line breaks because it was extracted mechanically from a table layout.

Extract every money transaction from this bank statement text.

First, locate the statement period stated at the top of the document (it will appear as a date range, such as a "from X to Y" style heading, often near the title). Use the start and end dates of this period to determine the correct year for every transaction date, since individual transaction dates in the table typically only show day and month, not the year.

Rule for inferring the year:
- If the statement period falls entirely within a single calendar year, use that year for all transactions.
- If the statement period spans two different calendar years (i.e. it starts in one year and ends in the next), compare each transaction's month to the period's start and end months: transactions whose month is closer to the period's start belong to the start year, and transactions whose month is closer to the period's end belong to the end year. For example, if a period starts in a late month of one year and ends in an early month of the following year, transactions in that late-year month take the start year, and transactions in that early-year month take the end year.

Also locate the statement's overall starting balance (the account balance at the very beginning of the period, before any listed transactions).

Return ONLY plain text in this exact format, no markdown, no code fences, no explanation.

First line must be exactly:
STARTING_BALANCE: <number>
where <number> is the statement's overall starting balance as a plain number with a period decimal separator (e.g. 1234.56).

Second line must be this exact header:
operation_date|description|debit|credit

Then one line per transaction, using these rules:
- operation_date: the date from the left-most "Date" column (do not use any "Valeur"/value-date column, even if present), formatted as YYYY-MM-DD using the inferred year
- description: full operation text, with any pipe characters removed if present, and internal line breaks replaced with a single space
- debit: the debit amount as a plain number with a period decimal separator (e.g. 3.99), or empty if not a debit
- credit: the credit amount as a plain number with a period decimal separator, or empty if not a credit

Process transactions in the exact order they appear in the document.

Do not skip any transaction, including small commission or fee lines.

Do not invent, merge, or reorder transactions that aren't clearly present in the text below.

--- STATEMENT TEXT START ---
{statement_text}
--- STATEMENT TEXT END ---
"""

# --- Step 3: run local inference ---
print(f"Loading model from {MODEL_PATH} ...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=os.cpu_count(),
    verbose=False,
)

try:
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8000,
    )
except Exception as e:
    print(f"Local inference failed: {e}")
    raise SystemExit(1)

output_text = response["choices"][0]["message"]["content"].strip()

# Defensively strip code fences in case the model adds them anyway
if output_text.startswith("```"):
    output_text = output_text.strip("`").strip()
    if output_text.lower().startswith("text"):
        output_text = output_text[4:].strip()

lines = [line for line in output_text.split("\n") if line.strip()]

if not lines or not lines[0].upper().startswith("STARTING_BALANCE"):
    print("ERROR: Model output did not start with STARTING_BALANCE line as expected.")
    print("--- Raw model output ---")
    print(output_text)
    raise SystemExit(1)

try:
    starting_balance = float(lines[0].split(":", 1)[1].strip())
except (IndexError, ValueError) as e:
    print(f"ERROR: Could not parse starting balance from '{lines[0]}': {e}")
    raise SystemExit(1)

header = lines[1].split("|")
data_lines = lines[2:]

# --- Step 4: parse rows and compute running balance deterministically ---
transactions = []
running_balance = starting_balance

for line in data_lines:
    fields = line.split("|")
    row = dict(zip(header, fields))

    debit_str = row.get("debit", "").strip()
    credit_str = row.get("credit", "").strip()
    debit = float(debit_str) if debit_str else 0.0
    credit = float(credit_str) if credit_str else 0.0

    row_starting_balance = running_balance
    running_balance = running_balance - debit + credit

    transactions.append(
        {
            "operation_date": row.get("operation_date", "").strip(),
            "description": row.get("description", "").strip(),
            "debit": debit_str,
            "credit": credit_str,
            "starting_balance": f"{row_starting_balance:.2f}",
            "ending_balance": f"{running_balance:.2f}",
        }
    )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(transactions, f, ensure_ascii=False, indent=2)

print(f"Saved {len(transactions)} transactions to {OUTPUT_PATH}")
print(f"Final computed ending balance: {running_balance:.2f}")
