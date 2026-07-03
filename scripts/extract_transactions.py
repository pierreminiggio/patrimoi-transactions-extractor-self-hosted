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


statement_text = extract_pdf_text(PDF_PATH)

if not statement_text.strip():
    print(
        "ERROR: Could not extract any text from the PDF. It may be a scanned "
        "image rather than a text-based PDF, which this self-hosted pipeline "
        "does not currently handle (no OCR step is included)."
    )
    raise SystemExit(1)

# --- Step 2: build the prompt ---
# The model is only asked to extract raw fields (date, description, debit,
# credit, and — if the document shows one — a per-transaction running
# balance). Everything requiring arithmetic, date math, sign conventions, or
# chronological ordering is done deterministically in Python afterward,
# because small local models have proven unreliable at all of those across
# a range of real statement formats (French and English, table-based and
# block-based, chronological and reverse-chronological).
prompt = f"""You are given the raw text extracted from a bank/financial account statement PDF, with the original layout preserved using spacing — columns are visually aligned using whitespace, the same way they'd look if you printed the PDF as plain text with a monospaced font. This spacing is meaningful and must be read carefully.

Extract every money transaction from this statement text. Statements come from different providers and look different: some are tables with a "Date | Description | Débit | Crédit" style layout (French), others list each transaction as its own block of a few lines with labels like "Incoming"/"Outgoing"/"Amount" (English), and others use different wording still. Apply the general principles below regardless of the exact wording used in this particular document.

CRITICAL — determining debit vs credit for each transaction, in order of preference:
1. If amounts appear in two separate columns with headers (in any language) roughly meaning "money out" and "money in" — e.g. "Débit"/"Crédit", "Debit"/"Credit", "Outgoing"/"Incoming", "Withdrawal"/"Deposit" — determine which column an amount belongs to by comparing its horizontal position (character offset from the start of the line) to the horizontal position of those header words. Do not guess based on what the transaction sounds like (e.g. do not assume "received"-sounding wording is automatically a credit) — the column position is the ground truth.
2. If amounts appear with an explicit sign or a single "Outgoing" value that is written as a negative number (e.g. "-23.56"), a negative/outgoing value is a debit and a positive/incoming value is a credit.
Do not put a value in both debit and credit for the same transaction — exactly one of the two must be filled in, the other left empty.

CRITICAL — statement balance: some statements state an explicit overall balance at specific points, e.g. "SOLDE DEBITEUR AU <date>" / "SOLDE CREDITEUR AU <date>" (French) or "balance as of <date>" / "opening balance" / "closing balance" (English) or similar wording in another language. If you find such statements, use the earliest one for STARTING_BALANCE and the latest (or most current) one for ENDING_BALANCE. In French, "DEBITEUR" means the account is overdrawn — record that figure as a NEGATIVE number even though it's printed as a plain positive amount (e.g. "SOLDE DEBITEUR AU 07.03.2025  9,60" means -9.60); "CREDITEUR" is a normal positive balance, record as printed.
If the document does NOT state an explicit overall starting or ending balance anywhere, but instead shows a running balance next to each individual transaction (e.g. an "Amount" column that is the balance immediately after that transaction), then write UNKNOWN for STARTING_BALANCE and/or ENDING_BALANCE instead of a number, and instead fill in the resulting_balance field (described below) for every transaction — a separate deterministic process will derive the overall starting/ending balance from those per-transaction values.

CRITICAL — some statements list transactions oldest-first, others list them newest-first. You do not need to figure out which, or reorder anything — just transcribe each transaction row once, in whatever order it appears in the source text. A separate deterministic process will sort everything chronologically afterward.

Also: the text may contain summary/total lines (e.g. "SOLDE DEBITEUR/CREDITEUR AU <date>", "TOTAL DES OPERATIONS", "TOTAL DES MOUVEMENTS", or similar wording) that restate a balance or a total rather than describing one specific transaction. Do NOT include these as transaction rows — skip them entirely (their balance values are already captured via STARTING_BALANCE/ENDING_BALANCE above, if present). Only genuine individual transactions (payments, transfers, fees, deposits, purchases, cashback, etc.) belong in the output rows.

Return ONLY plain text in this exact format, no markdown, no code fences, no explanation.

IMPORTANT: convert every amount to use a period as the decimal separator (e.g. "9.60", not "9,60"), and remove any thousands separators (spaces, periods, or commas used as thousands separators). Never output a comma inside a number in your output.

IMPORTANT: preserve normal spacing between words in the description field exactly as a human would write it (e.g. "VIR SCT INST RECU /DE PIERRE MINIGGIO", not "VIRSCTINSTRECU/DEPIERREMINIGGIO"). The source text may have extra or irregular spacing from table formatting — collapse multiple consecutive spaces into one, but always keep at least one space between separate words.

First line must be exactly:
STARTING_BALANCE: <number or UNKNOWN>

Second line must be exactly:
ENDING_BALANCE: <number or UNKNOWN>

Third line must be this exact header:
operation_date|description|debit|credit|resulting_balance

Then one line per transaction, using these rules:
- operation_date: the transaction's own date, copied EXACTLY as printed (e.g. "09.05", "07.03.2025", or "4 June 2026") — do not add, guess, or compute a year yourself, and do not reformat it; a separate deterministic process handles that. If a table has both a "Date" and a "Valeur"/value-date column, use "Date", not "Valeur".
- description: the transaction's own description text only. If it wraps onto one or more additional lines (continuation lines with no date at the start, often containing extra reference details), append that continuation text with a single space — do not create a separate transaction or drop it. Do not include a "Valeur" column date or other column's value inside the description. Remove any pipe characters if present.
- debit: the debit/outgoing amount as a plain positive number with a period decimal separator, or empty if this transaction is a credit
- credit: the credit/incoming amount as a plain positive number with a period decimal separator, or empty if this transaction is a debit
- resulting_balance: the account balance shown immediately after this specific transaction, if the document displays one per transaction (as plain positive number, period decimal separator) — leave this empty if the document does not show a per-transaction balance (e.g. because it instead has explicit STARTING_BALANCE/ENDING_BALANCE statements).

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


def parse_balance_line(line: str, label: str):
    """Parse a STARTING_BALANCE / ENDING_BALANCE line, treating the literal
    value UNKNOWN (the model uses this when the document has no explicit
    overall balance statement) as None rather than an error.
    """
    raw = line.split(":", 1)[1].strip()
    if raw.upper() == "UNKNOWN":
        return None
    try:
        value = parse_number(raw)
        if value is None:
            raise ValueError("empty value")
        return value
    except ValueError as e:
        print(f"ERROR: Could not parse {label} from '{line}': {e}")
        raise SystemExit(1)


starting_balance = parse_balance_line(lines[0], "starting balance")

if len(lines) < 2 or not lines[1].upper().startswith("ENDING_BALANCE"):
    print("ERROR: Model output did not include an ENDING_BALANCE line as expected.")
    print("--- Raw model output ---")
    print(output_text)
    raise SystemExit(1)

stated_ending_balance = parse_balance_line(lines[1], "ending balance")

header = lines[2].split("|")
data_lines = lines[3:]


# Defensive fallback: cross-check the sign of the starting/ending balance
# against the actual "SOLDE DEBITEUR/CREDITEUR" wording found in the
# extracted text (French-specific pattern), and correct it if the model
# got the sign convention wrong. No-ops harmlessly for documents that
# don't use this wording (e.g. English statements), or when the balance
# is unknown (None) at this point.
def enforce_balance_sign(value, keyword_match):
    if value is None or keyword_match is None:
        return value
    is_debiteur = keyword_match.group(1).upper() == "DEBITEUR"
    if is_debiteur and value > 0:
        return -value
    if not is_debiteur and value < 0:
        return -value
    return value


solde_pattern = re.compile(
    r"(?i)SOLDE\s+(DEBITEUR|CREDITEUR)\s+AU\s+(\d{1,2})[./-](\d{1,2})[./-](\d{4})"
)
solde_matches = list(solde_pattern.finditer(statement_text))
first_solde_match = solde_matches[0] if solde_matches else None
last_solde_match = solde_matches[-1] if solde_matches else None

if starting_balance is not None:
    corrected = enforce_balance_sign(starting_balance, first_solde_match)
    if corrected != starting_balance:
        print(
            f"Corrected starting balance sign: model gave {starting_balance:.2f}, "
            f"but the source text says '{first_solde_match.group(0)}', so using "
            f"{corrected:.2f} instead."
        )
        starting_balance = corrected

if stated_ending_balance is not None:
    corrected = enforce_balance_sign(stated_ending_balance, last_solde_match)
    if corrected != stated_ending_balance:
        print(
            f"Corrected ending balance sign: model gave {stated_ending_balance:.2f}, "
            f"but the source text says '{last_solde_match.group(0)}', so using "
            f"{corrected:.2f} instead."
        )
        stated_ending_balance = corrected


# --- Statement period + year inference (fully deterministic) ---
# The model is never asked to compute a transaction's year itself — day/
# month-only dates (e.g. BNP's "09.05") need the statement's real period to
# resolve, which we extract here from whichever explicit source is present
# (SOLDE lines, or a date-range heading in French or English). Statements
# where every date already includes its own year (e.g. Wise's "4 June
# 2026") don't need this at all; normalize_operation_date handles that case
# directly without consulting the period.
FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}
ENGLISH_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
ALL_MONTHS = {**FRENCH_MONTHS, **ENGLISH_MONTHS}


def get_period_from_solde_lines():
    if not first_solde_match or not last_solde_match:
        return None, None
    start = (int(first_solde_match.group(4)), int(first_solde_match.group(3)), int(first_solde_match.group(2)))
    end = (int(last_solde_match.group(4)), int(last_solde_match.group(3)), int(last_solde_match.group(2)))
    return start, end  # each a (year, month, day) tuple


def get_period_from_header_text():
    # Fallback for statements without SOLDE lines: look for a date-range
    # heading. Try the specific French "du X au Y" phrasing first (most
    # precise), then fall back to just taking the first two recognizable
    # "<day> <month name> <year>" dates in document order — this works
    # generically since the period heading always appears before any
    # transaction dates, and avoids trying to match a single combined
    # regex across text that may contain annotations like "[GMT+02:00]"
    # between the two dates.
    m = re.search(
        r"(?i)du\s+(\d{1,2})\s+(\w+)\s+(\d{4})\s+au\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
        statement_text,
    )
    if m:
        d1, mo1, y1, d2, mo2, y2 = m.groups()
        mo1_num = ALL_MONTHS.get(mo1.lower())
        mo2_num = ALL_MONTHS.get(mo2.lower())
        if mo1_num is not None and mo2_num is not None:
            return (int(y1), mo1_num, int(d1)), (int(y2), mo2_num, int(d2))

    candidates = []
    for dm in re.finditer(r"(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", statement_text):
        d, mo, y = dm.groups()
        mo_num = ALL_MONTHS.get(mo.lower())
        if mo_num is not None:
            candidates.append((int(y), mo_num, int(d)))
        if len(candidates) >= 2:
            break
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    return None, None


period_start, period_end = get_period_from_solde_lines()
if period_start is None or period_end is None:
    period_start, period_end = get_period_from_header_text()

if period_start is None or period_end is None:
    print(
        "NOTE: could not determine the statement's date period from an "
        "explicit balance statement or date-range heading. This is fine as "
        "long as every transaction date already includes its own year; "
        "day.month-only dates would be left un-converted otherwise."
    )


def infer_year(month: int):
    if period_start is None or period_end is None:
        return None
    start_year, start_month = period_start[0], period_start[1]
    end_year, end_month = period_end[0], period_end[1]
    if start_year == end_year:
        return start_year

    def month_distance(m1, m2):
        d = abs(m1 - m2)
        return min(d, 12 - d)

    if month_distance(month, start_month) <= month_distance(month, end_month):
        return start_year
    return end_year


def normalize_operation_date(raw: str) -> str:
    raw = raw.strip()
    # Already in YYYY-MM-DD format — trust it as-is.
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    # DD.MM.YYYY or DD/MM/YYYY — year already present, just reformat.
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$", raw)
    if m:
        day, month, year = m.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    # Textual date with its own explicit year, e.g. "4 June 2026" or "4 juin 2026".
    m = re.match(r"^(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})$", raw)
    if m:
        day, month_name, year = m.groups()
        month_num = ALL_MONTHS.get(month_name.lower())
        if month_num is not None:
            return f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
    # DD.MM or DD/MM — no year, infer it from the statement period.
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})$", raw)
    if m:
        day, month = m.groups()
        year = infer_year(int(month))
        if year is None:
            print(f"WARNING: could not infer a year for date '{raw}'; leaving it un-converted.")
            return raw
        return f"{year:04d}-{int(month):02d}-{int(day):02d}"
    # Unrecognized format — leave as-is rather than guessing.
    print(f"WARNING: unrecognized date format '{raw}'; leaving it un-converted.")
    return raw


# --- Debit/credit column detection (fully deterministic) ---
# Different providers label these columns differently and some use fixed
# column positions (BNP's "Débit"/"Crédit") while others use an explicit
# sign on a single amount (Wise's "-23.56" for outgoing). We try a handful
# of common label pairs for the position-based approach; the amount_pattern
# below also allows an optional leading minus sign so a negative "Outgoing"
# value can still be located and cross-checked.
COLUMN_LABEL_PAIRS = [
    (r"D[ée]bit", r"Cr[ée]dit"),
    (r"Outgoing", r"Incoming"),
    (r"Withdrawal", r"Deposit"),
]


def find_header_column_positions(text):
    for line in text.split("\n"):
        for debit_label, credit_label in COLUMN_LABEL_PAIRS:
            debit_match = re.search(debit_label, line, re.IGNORECASE)
            credit_match = re.search(credit_label, line, re.IGNORECASE)
            if debit_match and credit_match:
                return debit_match.start(), credit_match.start()
    return None, None


DEBIT_COL, CREDIT_COL = find_header_column_positions(statement_text)
statement_lines = statement_text.split("\n")
_line_search_cursor = 0

# Detect whether this document encodes debits with an explicit minus sign
# anywhere at all (e.g. Wise's "-23.56"). If it does, absence of a sign is
# just as informative as its presence (positive = credit) for this
# document, and both are more reliable than character-position comparison,
# which assumes a fixed-width column grid that doesn't hold for every
# layout (it doesn't for Wise, where numbers sit right after a variable-
# length description rather than in a true fixed column). If the document
# never uses a minus sign anywhere (e.g. BNP, all-positive amounts), this
# stays False and position comparison remains the only available signal.
DOCUMENT_USES_SIGNED_AMOUNTS = bool(re.search(r"-\d+[.,]\d{2}", statement_text))


def locate_amount_column(description: str, amount_str: str):
    """Find the source line for this transaction (by matching a chunk of
    its description text) and determine which column the amount actually
    sits under, based on its character position relative to the detected
    debit/credit-style header positions. Returns 'debit', 'credit', or None
    if the line/amount/headers can't be confidently located (in which case
    the model's original answer is left untouched rather than guessed at).
    """
    global _line_search_cursor
    if DEBIT_COL is None or CREDIT_COL is None or not amount_str:
        return None
    desc_key = re.sub(r"\s+", "", description)[:12].upper()
    if not desc_key:
        return None
    try:
        amount_pattern = r"-?" + re.escape(f"{float(amount_str):.2f}").replace(r"\.", "[.,]")
    except ValueError:
        return None
    for i in range(_line_search_cursor, len(statement_lines)):
        line = statement_lines[i]
        line_key = re.sub(r"\s+", "", line).upper()
        if desc_key and desc_key in line_key:
            _line_search_cursor = i + 1
            matches = list(re.finditer(amount_pattern, line))
            if not matches:
                return None
            # A description can coincidentally contain the same number as
            # the transaction amount (e.g. Wise's "Card transaction of
            # 23.56 EUR issued by..." repeats the amount in prose before
            # the actual column value appears later in the line). Picking
            # the first match can grab that coincidental mention instead of
            # the real column-aligned value, so pick whichever match sits
            # closest to either detected column position instead.
            best_match = min(
                matches,
                key=lambda m: min(abs(m.start() - DEBIT_COL), abs(m.start() - CREDIT_COL)),
            )
            # An explicit minus sign directly on the matched number (e.g.
            # Wise's "-23.56" for outgoing amounts) is a much more reliable
            # signal than character-position comparison, which assumes a
            # fixed-width column grid — that assumption held for BNP's
            # bordered table but doesn't hold for layouts where numbers sit
            # right after a variable-length description rather than in a
            # true fixed column (as seen with Wise). If this document uses
            # signed amounts anywhere, trust the sign fully (its absence on
            # a match means credit, not just "no evidence either way") and
            # skip position comparison entirely. Only documents that never
            # use a sign at all (e.g. BNP) fall back to position.
            if DOCUMENT_USES_SIGNED_AMOUNTS:
                return "debit" if best_match.group(0).startswith("-") else "credit"
            amount_pos = best_match.start()
            return "debit" if abs(amount_pos - DEBIT_COL) < abs(amount_pos - CREDIT_COL) else "credit"
    return None


# --- Step 4: parse every row (pass 1, unordered) ---
# We no longer trust the model's row order to already be chronological —
# some statements (e.g. Wise) list transactions newest-first, others
# oldest-first. Every row is parsed and date-normalized first; sorting and
# running-balance computation happen afterward as a second pass.
parsed_rows = []

for line in data_lines:
    fields = line.split("|")
    row = dict(zip(header, fields))

    description = row.get("description", "").strip()
    # Defensive fallback: strip a stray trailing "Valeur" column date
    # (format DD.MM) if the model still leaks one onto the end of the
    # description despite the prompt instruction not to include it.
    description = re.sub(r"\s+\d{2}\.\d{2}$", "", description).strip()

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

    # Cross-check the column assignment against the raw text rather than
    # trusting the model outright.
    amount_str = debit_str or credit_str
    detected_column = locate_amount_column(description, amount_str)
    if detected_column == "credit" and debit_str and not credit_str:
        print(
            f"Corrected debit/credit column for '{description}': model said "
            f"debit, but its position in the source text lines up with the "
            f"credit column."
        )
        debit_str, credit_str = "", debit_str
        debit, credit = 0.0, debit
    elif detected_column == "debit" and credit_str and not debit_str:
        print(
            f"Corrected debit/credit column for '{description}': model said "
            f"credit, but its position in the source text lines up with the "
            f"debit column."
        )
        credit_str, debit_str = "", credit_str
        credit, debit = 0.0, credit

    resulting_balance_str = row.get("resulting_balance", "").strip()
    try:
        resulting_balance = parse_number(resulting_balance_str)
    except ValueError:
        resulting_balance = None

    parsed_rows.append(
        {
            "operation_date": normalize_operation_date(row.get("operation_date", "").strip()),
            "description": description,
            "debit_str": debit_str,
            "credit_str": credit_str,
            "debit": debit,
            "credit": credit,
            "resulting_balance": resulting_balance,
        }
    )

# --- Ensure chronological (oldest first) order, regardless of source order ---
# Some statements list transactions oldest-first, others newest-first. A
# plain sort-by-date would fix the overall direction but loses same-day
# relative ordering (multiple transactions sharing one date get collapsed
# to sort-stable order, which is only correct if the source already
# happened to list them oldest-first within that day too). Instead, detect
# the document's overall direction and reverse the whole list if needed —
# this preserves the model's true relative transcription order (including
# same-day order) rather than discarding it.
def _looks_ascending(rows):
    iso_dates = [r["operation_date"] for r in rows if re.match(r"^\d{4}-\d{2}-\d{2}$", r["operation_date"])]
    if len(iso_dates) < 2:
        return True  # not enough evidence to tell; assume already correct
    return iso_dates[0] <= iso_dates[-1]


if not _looks_ascending(parsed_rows):
    parsed_rows.reverse()

# --- Determine the starting balance if the document didn't state one explicitly ---
if starting_balance is None:
    anchor = next((r for r in parsed_rows if r["resulting_balance"] is not None), None)
    if anchor is not None:
        starting_balance = anchor["resulting_balance"] - anchor["credit"] + anchor["debit"]
        print(
            f"No explicit starting balance found in the document; derived "
            f"{starting_balance:.2f} from the earliest transaction's own "
            f"reported balance ({anchor['resulting_balance']:.2f}) and its "
            f"debit/credit."
        )
    else:
        starting_balance = 0.0
        print(
            "WARNING: could not determine a starting balance from either an "
            "explicit balance statement or a per-transaction running "
            "balance. Defaulting to 0.00 — starting_balance/ending_balance "
            "in the output will reflect only the net change across "
            "transactions, not the real account balance."
        )

if stated_ending_balance is None:
    anchor = next((r for r in reversed(parsed_rows) if r["resulting_balance"] is not None), None)
    if anchor is not None:
        stated_ending_balance = anchor["resulting_balance"]
        print(
            f"No explicit ending balance found in the document; using the "
            f"most recent transaction's own reported balance "
            f"({stated_ending_balance:.2f}) instead."
        )

# --- Step 5: compute running balance deterministically, in chronological order ---
transactions = []
running_balance = starting_balance

for r in parsed_rows:
    row_starting_balance = running_balance
    running_balance = running_balance - r["debit"] + r["credit"]
    transactions.append(
        {
            "operation_date": r["operation_date"],
            "description": r["description"],
            "debit": f"{r['debit']:.2f}" if r["debit_str"] else "",
            "credit": f"{r['credit']:.2f}" if r["credit_str"] else "",
            "starting_balance": f"{row_starting_balance:.2f}",
            "ending_balance": f"{running_balance:.2f}",
        }
    )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(transactions, f, ensure_ascii=False, indent=2)

print(f"Saved {len(transactions)} transactions to {OUTPUT_PATH}")
print(f"Computed ending balance: {running_balance:.2f}")

if stated_ending_balance is not None:
    print(f"Statement's stated/derived ending balance: {stated_ending_balance:.2f}")
    if abs(running_balance - stated_ending_balance) > 0.01:
        print(
            "WARNING: computed ending balance does not match the statement's "
            "stated ending balance. This usually means a transaction was "
            "missed, misread, or a debit/credit was misclassified — review "
            f"{OUTPUT_PATH} against the original PDF before trusting it."
        )
else:
    print("No ending balance could be determined from the document to validate against.")
