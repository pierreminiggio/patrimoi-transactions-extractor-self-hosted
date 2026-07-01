import json
import os
import re
import sys

import pdfplumber
from llama_cpp import Llama

# --- Configuration via environment / args ---
MODEL_PATH = os.environ.get("MODEL_PATH", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
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
            # layout=True preserves the original spatial positions of words
            # (via whitespace padding), so table columns stay visually
            # separated instead of being collapsed into one run-on string.
            # This matters a lot here: without it, "SOLDE DEBITEUR AU
            # 07.03.2025" becomes "SOLDEDEBITEURAU07.03.2025", and the
            # debit/credit columns become impossible to tell apart.
            #
            # x_tolerance controls how close two characters need to be
            # before pdfplumber treats them as part of the same "word" —
            # this happens *before* layout mode runs, so if the statement's
            # table uses tight kerning, the default tolerance can merge
            # genuinely separate words into a single token that no amount
            # of layout padding can then split apart. Lowering it makes
            # word-boundary detection stricter.
            text = page.extract_text(layout=True, x_tolerance=1) or ""
            pages_text.append(text)
    full_text = "\n".join(pages_text)
    # NOTE: do not collapse internal whitespace runs here. The horizontal
    # position of a number (e.g. how far right "15,00" sits relative to the
    # "Débit"/"Crédit" column headers) is the ONLY signal that tells us
    # which column it belongs to, since these statements have no other
    # per-row column markers. Collapsing long space runs down to a fixed
    # width destroys that signal and makes debit/credit indistinguishable
    # (this was tried and caused misclassified transactions). Trailing
    # whitespace per line is harmless to trim since it carries no column
    # information.
    full_text = "\n".join(line.rstrip() for line in full_text.split("\n"))
    return full_text


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
prompt = f"""You are given the raw text extracted from a bank statement PDF, with the original table layout preserved using spacing — columns are visually aligned using whitespace, the same way they'd look if you printed the PDF as plain text with a monospaced font. This spacing is meaningful and must be read carefully.

Extract every money transaction from this bank statement text.

CRITICAL — reading the Débit vs Crédit columns: the header row contains column names (e.g. "Date", "Nature des opérations", "Valeur", "Débit", "Crédit") at specific horizontal positions. For every amount in a transaction row, determine whether it is a debit or a credit by comparing its horizontal position (how many characters from the start of the line) to the horizontal position of the "Débit" and "Crédit" words in the header row — an amount is a debit if it lines up under "Débit", and a credit if it lines up under "Crédit". Do not guess based on typical transaction meaning (e.g. do not assume a wording like "RECU"/"received" is automatically a credit) — always verify against the actual column position in the text, since that is the ground truth. A single row will have a value in only one of the two columns, never both.

First, locate the statement period stated at the top of the document (it will appear as a date range, such as a "from X to Y" style heading, often near the title). Use the start and end dates of this period to determine the correct year for every transaction date, since individual transaction dates in the table typically only show day and month, not the year.

Rule for inferring the year:
- If the statement period falls entirely within a single calendar year, use that year for all transactions.
- If the statement period spans two different calendar years (i.e. it starts in one year and ends in the next), compare each transaction's month to the period's start and end months: transactions whose month is closer to the period's start belong to the start year, and transactions whose month is closer to the period's end belong to the end year. For example, if a period starts in a late month of one year and ends in an early month of the following year, transactions in that late-year month take the start year, and transactions in that early-year month take the end year.

Also locate the statement's overall starting balance (the account balance at the very beginning of the period, before any listed transactions) and its overall ending balance (the account balance at the very end of the period, after all listed transactions).

IMPORTANT: the statement text may contain lines like "SOLDE DEBITEUR AU <date>" or "SOLDE CREDITEUR AU <date>" (or similar "balance as of <date>" wording), or a "TOTAL DES OPERATIONS" (or "TOTAL DES MOUVEMENTS") row summing all debits and credits for the period. These are all summary lines, NOT transactions — they restate a balance or a total rather than describing a single instance of money moving in or out. Do not include any of these lines as transactions in your output. However, the FIRST "SOLDE..." line gives you the STARTING_BALANCE value, and the LAST "SOLDE..." line (or the statement's explicit closing balance figure) gives you the ENDING_BALANCE value — extract both of these numbers from those lines, you just don't repeat the lines themselves as transaction rows. The rest of the output should contain only genuine transaction rows (payments, transfers, fees, deposits, purchases, etc.).

IMPORTANT — sign convention: "SOLDE DEBITEUR" means the account is overdrawn (the customer owes the bank money), so this figure must be recorded as a NEGATIVE number even though it's printed as a plain positive amount on the statement. For example, "SOLDE DEBITEUR AU 07.03.2025  9,60" means STARTING_BALANCE is -9.60, not 9.60. "SOLDE CREDITEUR" means a normal positive balance, so record that figure as-is (positive). Apply this same sign rule to whichever of STARTING_BALANCE or ENDING_BALANCE comes from a "DEBITEUR" line.

Return ONLY plain text in this exact format, no markdown, no code fences, no explanation.

IMPORTANT: the source document is a French bank statement and will show amounts with a comma as decimal separator (e.g. "9,60"). You must convert every amount to use a period as the decimal separator instead (e.g. "9.60"), and remove any thousands separators (spaces or periods). Never output a comma inside a number.

IMPORTANT: dates in the source document may appear as DD.MM.YYYY or DD/MM (without a year). You must always convert these to YYYY-MM-DD format using the inferred year — never output a date in its original DD.MM.YYYY form.

IMPORTANT: preserve the normal spacing between words in the description field exactly as a human would write it (e.g. "VIR SCT INST RECU /DE PIERRE MINIGGIO", not "VIRSCTINSTRECU/DEPIERREMINIGGIO"). The source text may have extra or irregular spacing from table formatting — collapse multiple consecutive spaces into one, but always keep at least one space between separate words.

First line must be exactly:
STARTING_BALANCE: <number>
where <number> is the statement's overall starting balance as a plain number with a period decimal separator (e.g. 1234.56).

Second line must be exactly:
ENDING_BALANCE: <number>
where <number> is the statement's overall ending balance, same format.

Third line must be this exact header:
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

def parse_number(raw: str):
    """Parse a number that may use either '.' or ',' as the decimal
    separator, and may include thousands separators (spaces, apostrophes,
    or the other punctuation mark), despite the prompt asking for a plain
    period-decimal number. Small local models don't always follow that
    instruction consistently, so we normalize defensively instead of
    trusting the format.
    """
    s = raw.strip().replace("\u00a0", "").replace(" ", "").replace("'", "")
    if not s:
        return None
    if "," in s and "." in s:
        # Whichever separator appears last is the decimal separator;
        # the other is a thousands separator and gets dropped.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    return float(s)


try:
    starting_balance = parse_number(lines[0].split(":", 1)[1])
    if starting_balance is None:
        raise ValueError("empty value")
except (IndexError, ValueError) as e:
    print(f"ERROR: Could not parse starting balance from '{lines[0]}': {e}")
    raise SystemExit(1)

if len(lines) < 2 or not lines[1].upper().startswith("ENDING_BALANCE"):
    print("ERROR: Model output did not include an ENDING_BALANCE line as expected.")
    print("--- Raw model output ---")
    print(output_text)
    raise SystemExit(1)

try:
    stated_ending_balance = parse_number(lines[1].split(":", 1)[1])
    if stated_ending_balance is None:
        raise ValueError("empty value")
except (IndexError, ValueError) as e:
    print(f"ERROR: Could not parse ending balance from '{lines[1]}': {e}")
    raise SystemExit(1)

header = lines[2].split("|")
data_lines = lines[3:]


# Defensive fallback: cross-check the sign of the starting/ending balance
# against the actual "SOLDE DEBITEUR/CREDITEUR" wording found in the
# extracted text, and correct it if the model got the sign convention wrong.
def enforce_balance_sign(value: float, keyword_match) -> float:
    if keyword_match is None:
        return value
    is_debiteur = keyword_match.group(1).upper() == "DEBITEUR"
    if is_debiteur and value > 0:
        return -value
    if not is_debiteur and value < 0:
        return -value
    return value


solde_matches = list(re.finditer(r"(?i)SOLDE\s+(DEBITEUR|CREDITEUR)\s+AU\b", statement_text))
first_solde_match = solde_matches[0] if solde_matches else None
last_solde_match = solde_matches[-1] if solde_matches else None

corrected_starting_balance = enforce_balance_sign(starting_balance, first_solde_match)
if corrected_starting_balance != starting_balance:
    print(
        f"Corrected starting balance sign: model gave {starting_balance:.2f}, "
        f"but the source text says '{first_solde_match.group(0)}', so using "
        f"{corrected_starting_balance:.2f} instead."
    )
    starting_balance = corrected_starting_balance

corrected_ending_balance = enforce_balance_sign(stated_ending_balance, last_solde_match)
if corrected_ending_balance != stated_ending_balance:
    print(
        f"Corrected ending balance sign: model gave {stated_ending_balance:.2f}, "
        f"but the source text says '{last_solde_match.group(0)}', so using "
        f"{corrected_ending_balance:.2f} instead."
    )
    stated_ending_balance = corrected_ending_balance

# --- Step 4: parse rows and compute running balance deterministically ---
transactions = []
running_balance = starting_balance

for line in data_lines:
    fields = line.split("|")
    row = dict(zip(header, fields))

    description = row.get("description", "").strip()

    # Defensive fallback: even with the prompt instruction, a small model
    # can occasionally still echo a "SOLDE DEBITEUR/CREDITEUR AU ..." balance
    # line, or a "TOTAL DES OPERATIONS" summary line, as if it were a
    # transaction row. Both the starting and ending balance values were
    # already captured above from the STARTING_BALANCE / ENDING_BALANCE
    # lines, so it's safe to drop any stray summary line here without
    # losing that information.
    if re.match(r"(?i)^SOLDE\s+(DEBITEUR|CREDITEUR)\s+AU\b", description):
        print(f"Skipping balance-summary line mistakenly included as a transaction: '{description}'")
        continue
    if re.match(r"(?i)^TOTAL\s+DES\s+(OPERATIONS|MOUVEMENTS)\b", description):
        print(f"Skipping totals-summary line mistakenly included as a transaction: '{description}'")
        continue

    debit_str = row.get("debit", "").strip()
    credit_str = row.get("credit", "").strip()
    try:
        debit = parse_number(debit_str) or 0.0
        credit = parse_number(credit_str) or 0.0
    except ValueError as e:
        print(f"ERROR: Could not parse debit/credit on line '{line}': {e}")
        raise SystemExit(1)

    row_starting_balance = running_balance
    running_balance = running_balance - debit + credit

    transactions.append(
        {
            "operation_date": row.get("operation_date", "").strip(),
            "description": description,
            "debit": f"{debit:.2f}" if debit_str else "",
            "credit": f"{credit:.2f}" if credit_str else "",
            "starting_balance": f"{row_starting_balance:.2f}",
            "ending_balance": f"{running_balance:.2f}",
        }
    )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(transactions, f, ensure_ascii=False, indent=2)

print(f"Saved {len(transactions)} transactions to {OUTPUT_PATH}")
print(f"Computed ending balance: {running_balance:.2f}")
print(f"Statement's stated ending balance: {stated_ending_balance:.2f}")

if abs(running_balance - stated_ending_balance) > 0.01:
    print(
        "WARNING: computed ending balance does not match the statement's "
        "stated ending balance. This usually means a transaction was missed, "
        "misread, or a debit/credit was misclassified — review "
        f"{OUTPUT_PATH} against the original PDF before trusting it."
    )
