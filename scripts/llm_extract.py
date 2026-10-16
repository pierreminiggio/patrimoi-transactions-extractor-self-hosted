"""Everything to do with the local LLM: prompt construction, lazy model
loading (never touched unless a deterministic path couldn't handle the
input), and parsing the model's raw text output into rows.

This is deliberately source-agnostic: `source_text` can be text extracted
from a PDF, or the raw content of an unrecognized CSV/plain-text export.
The model's job is the same either way — locate and transcribe transaction
fields — everything requiring arithmetic, date math, or ordering is done
afterward in Python (see common.py).
"""

import os
import re

PROMPT_TEMPLATE = """You are given raw text describing a bank/financial account statement (extracted from a PDF, or provided directly as CSV/plain text), with the original layout preserved using spacing where applicable — columns may be visually aligned using whitespace, the same way they'd look if you printed the source as plain text with a monospaced font. This spacing is meaningful and must be read carefully.

Extract every money transaction from this statement text. Statements come from different providers and look different: some are tables with a "Date | Description | Débit | Crédit" style layout (French), others list each transaction as its own block of a few lines with labels like "Incoming"/"Outgoing"/"Amount" (English), others are comma-separated CSV rows with a header line, and others use different wording still. Apply the general principles below regardless of the exact wording or format used in this particular document.

CRITICAL — determining debit vs credit for each transaction, in order of preference:
1. If amounts appear in two separate columns with headers (in any language) roughly meaning "money out" and "money in" — e.g. "Débit"/"Crédit", "Debit"/"Credit", "Outgoing"/"Incoming", "Withdrawal"/"Deposit" — determine which column an amount belongs to by comparing its horizontal position (character offset from the start of the line) to the horizontal position of those header words. Do not guess based on what the transaction sounds like (e.g. do not assume "received"-sounding wording is automatically a credit) — the column position is the ground truth.
2. If amounts appear with an explicit sign or a single "Outgoing"/"Amount" value that is written as a negative number (e.g. "-23.56"), a negative/outgoing value is a debit and a positive/incoming value is a credit.
Do not put a value in both debit and credit for the same transaction — exactly one of the two must be filled in, the other left empty.

CRITICAL — statement balance: some statements state an explicit overall balance at specific points, e.g. "SOLDE DEBITEUR AU <date>" / "SOLDE CREDITEUR AU <date>" (French) or "balance as of <date>" / "opening balance" / "closing balance" (English) or similar wording in another language. If you find such statements, use the earliest one for STARTING_BALANCE and the latest (or most current) one for ENDING_BALANCE. In French, "DEBITEUR" means the account is overdrawn — record that figure as a NEGATIVE number even though it's printed as a plain positive amount (e.g. "SOLDE DEBITEUR AU 07.03.2025  9,60" means -9.60); "CREDITEUR" is a normal positive balance, record as printed.
If the document does NOT state an explicit overall starting or ending balance anywhere, but instead shows a running balance next to each individual transaction (e.g. an "Amount"/"Balance" column that is the balance immediately after that transaction), then write UNKNOWN for STARTING_BALANCE and/or ENDING_BALANCE instead of a number, and instead fill in the resulting_balance field (described below) for every transaction — a separate deterministic process will derive the overall starting/ending balance from those per-transaction values.
If neither an explicit balance statement nor a per-transaction running balance exists anywhere in the document (e.g. a plain CSV export with just dates/descriptions/amounts), write UNKNOWN for both STARTING_BALANCE and ENDING_BALANCE, and leave resulting_balance empty for every row.

CRITICAL — some statements list transactions oldest-first, others list them newest-first. You do not need to figure out which, or reorder anything — just transcribe each transaction row once, in whatever order it appears in the source text. A separate deterministic process will sort everything chronologically afterward.

Also: the text may contain summary/total lines (e.g. "SOLDE DEBITEUR/CREDITEUR AU <date>", "TOTAL DES OPERATIONS", "TOTAL DES MOUVEMENTS", or similar wording) that restate a balance or a total rather than describing one specific transaction. Do NOT include these as transaction rows — skip them entirely (their balance values are already captured via STARTING_BALANCE/ENDING_BALANCE above, if present). Only genuine individual transactions (payments, transfers, fees, deposits, purchases, cashback, etc.) belong in the output rows.

Return ONLY plain text in this exact format, no markdown, no code fences, no explanation.

IMPORTANT: convert every amount to use a period as the decimal separator (e.g. "9.60", not "9,60"), and remove any thousands separators (spaces, periods, or commas used as thousands separators). Never output a comma inside a number in your output.

IMPORTANT: preserve normal spacing between words in the description field exactly as a human would write it (e.g. "VIR SCT INST RECU /DE PIERRE MINIGGIO", not "VIRSCTINSTRECU/DEPIERREMINIGGIO"). The source text may have extra or irregular spacing from table formatting — collapse multiple consecutive spaces into one, but always keep at least one space between separate words. If the source is CSV, combine the relevant descriptive columns (e.g. counterparty name and payment reference) into one natural-language description, separated by " - ".

First line must be exactly:
STARTING_BALANCE: <number or UNKNOWN>

Second line must be exactly:
ENDING_BALANCE: <number or UNKNOWN>

Third line must be this exact header:
operation_date|description|debit|credit|resulting_balance

Then one line per transaction, using these rules:
- operation_date: the transaction's own date, copied EXACTLY as printed (e.g. "09.05", "07.03.2025", "2024-03-05", or "4 June 2026") — do not add, guess, or compute a year yourself, and do not reformat it; a separate deterministic process handles that. If a table has both a "Date" and a "Valeur"/value-date column, use "Date", not "Valeur".
- description: the transaction's own description text only. If it wraps onto one or more additional lines (continuation lines with no date at the start, often containing extra reference details), append that continuation text with a single space — do not create a separate transaction or drop it. Do not include a "Valeur" column date or other column's value inside the description. Remove any pipe characters if present.
- debit: the debit/outgoing amount as a plain positive number with a period decimal separator, or empty if this transaction is a credit
- credit: the credit/incoming amount as a plain positive number with a period decimal separator, or empty if this transaction is a debit
- resulting_balance: the account balance shown immediately after this specific transaction, if the document displays one per transaction (as plain positive number, period decimal separator) — leave this empty otherwise.

Do not skip any transaction, including small commission or fee lines.

Do not invent, merge, or reorder transactions that aren't clearly present in the text below.

--- STATEMENT TEXT START ---
{statement_text}
--- STATEMENT TEXT END ---
"""


def build_prompt(statement_text: str) -> str:
    return PROMPT_TEMPLATE.format(statement_text=statement_text)


_llm_singleton = None


def _get_llm(model_path: str, n_ctx: int):
    global _llm_singleton
    if _llm_singleton is None:
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}.")
            raise SystemExit(1)
        # Imported lazily too, so environments that only ever hit a
        # deterministic CSV path don't need llama-cpp-python installed.
        from llama_cpp import Llama

        print(f"Loading model from {model_path} ...")
        _llm_singleton = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=os.cpu_count(),
            verbose=False,
        )
    return _llm_singleton


def run_llm(source_text: str, model_path: str, n_ctx: int) -> str:
    llm = _get_llm(model_path, n_ctx)
    prompt = build_prompt(source_text)
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
    if output_text.startswith("```"):
        output_text = output_text.strip("`").strip()
        if output_text.lower().startswith("text"):
            output_text = output_text[4:].strip()
    return output_text


# --- Debit/credit column detection (fully deterministic, applied to the
# raw source text after the model has transcribed rows) ---
COLUMN_LABEL_PAIRS = [
    (r"D[ée]bit", r"Cr[ée]dit"),
    (r"Outgoing", r"Incoming"),
    (r"Withdrawal", r"Deposit"),
]


def _find_header_column_positions(text):
    for line in text.split("\n"):
        for debit_label, credit_label in COLUMN_LABEL_PAIRS:
            debit_match = re.search(debit_label, line, re.IGNORECASE)
            credit_match = re.search(credit_label, line, re.IGNORECASE)
            if debit_match and credit_match:
                return debit_match.start(), credit_match.start()
    return None, None


_DATE_FIELD_RE = re.compile(r"^\d{1,2}[./-]\d{1,2}([./-]\d{4})?$|^\d{4}-\d{2}-\d{2}$|^\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4}$")
_AMOUNT_FIELD_RE = re.compile(r"^-?\d[\d\s.,']*$")


def _realign_row_fields(fields, header):
    """Small models occasionally emit a row with the right content but the
    wrong shape (e.g. a spurious leading number that shifts every field one
    slot to the right). Re-anchor on the two fields that can be recognized
    unambiguously — the date and the description — and read the amounts
    from what follows the description. Anything before the date is dropped:
    it is not part of the row, and guessing what it meant (a balance? a
    total line leaking in?) would risk corrupting the data.

    Returns a dict keyed by `header`, or None if the row can't be anchored.
    """
    fields = [f.strip() for f in fields]
    desc_idx = next(
        (i for i, f in enumerate(fields) if re.search(r"[A-Za-zÀ-ÿ]{2,}", f) and not _DATE_FIELD_RE.match(f)),
        None,
    )
    if desc_idx is None:
        return None
    date_idx = next((i for i in range(desc_idx - 1, -1, -1) if _DATE_FIELD_RE.match(fields[i])), None)
    if date_idx is None:
        return None

    tail = fields[desc_idx + 1:]
    if any(f and not _AMOUNT_FIELD_RE.match(f) for f in tail):
        return None
    names = ["debit", "credit", "resulting_balance"]
    if len(tail) > len(names) and any(tail[len(names):]):
        return None
    tail = (tail + [""] * len(names))[: len(names)]
    return {
        "operation_date": fields[date_idx],
        "description": fields[desc_idx],
        **dict(zip(names, tail)),
    }


def _row_is_well_formed(fields, header):
    if len(fields) != len(header):
        return False
    row = dict(zip(header, (f.strip() for f in fields)))
    # Deliberately lenient about date/description content (unusual-but-valid
    # rows must keep taking the normal path); only the shape and the numeric
    # columns are checked, since that is what a shifted row breaks.
    return all(
        not row.get(k) or _AMOUNT_FIELD_RE.match(row[k])
        for k in ("debit", "credit", "resulting_balance")
    )


def parse_llm_output(output_text: str, source_text: str):
    """Parse the model's raw text output into parsed_rows plus
    starting/ending balance, cross-checking debit/credit assignment and
    balance signs against the raw source text rather than trusting the
    model outright.

    Returns (parsed_rows, starting_balance, stated_ending_balance).
    """
    from common import ALL_MONTHS, normalize_operation_date, parse_number

    lines = [line for line in output_text.split("\n") if line.strip()]

    if not lines or not lines[0].upper().startswith("STARTING_BALANCE"):
        print("ERROR: Model output did not start with STARTING_BALANCE line as expected.")
        print("--- Raw model output ---")
        print(output_text)
        raise SystemExit(1)

    def parse_balance_line(line: str, label: str):
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
    # against the actual "SOLDE DEBITEUR/CREDITEUR" wording found in the raw
    # text (French-specific pattern). No-ops harmlessly for documents that
    # don't use this wording.
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
    solde_matches = list(solde_pattern.finditer(source_text))
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
    def get_period_from_solde_lines():
        if not first_solde_match or not last_solde_match:
            return None, None
        start = (int(first_solde_match.group(4)), int(first_solde_match.group(3)), int(first_solde_match.group(2)))
        end = (int(last_solde_match.group(4)), int(last_solde_match.group(3)), int(last_solde_match.group(2)))
        return start, end

    def get_period_from_header_text():
        m = re.search(
            r"(?i)du\s+(\d{1,2})\s+(\w+)\s+(\d{4})\s+au\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
            source_text,
        )
        if m:
            d1, mo1, y1, d2, mo2, y2 = m.groups()
            mo1_num = ALL_MONTHS.get(mo1.lower())
            mo2_num = ALL_MONTHS.get(mo2.lower())
            if mo1_num is not None and mo2_num is not None:
                return (int(y1), mo1_num, int(d1)), (int(y2), mo2_num, int(d2))

        candidates = []
        for dm in re.finditer(r"(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", source_text):
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

    DEBIT_COL, CREDIT_COL = _find_header_column_positions(source_text)
    source_lines = source_text.split("\n")
    line_search_cursor = [0]

    DOCUMENT_USES_SIGNED_AMOUNTS = bool(re.search(r"-\d+[.,]\d{2}", source_text))

    def locate_amount_column(description: str, amount_str: str):
        if DEBIT_COL is None or CREDIT_COL is None or not amount_str:
            return None
        desc_key = re.sub(r"\s+", "", description)[:12].upper()
        if not desc_key:
            return None
        try:
            amount_pattern = r"-?" + re.escape(f"{float(amount_str):.2f}").replace(r"\.", "[.,]")
        except ValueError:
            return None
        for i in range(line_search_cursor[0], len(source_lines)):
            line = source_lines[i]
            line_key = re.sub(r"\s+", "", line).upper()
            if desc_key and desc_key in line_key:
                line_search_cursor[0] = i + 1
                matches = list(re.finditer(amount_pattern, line))
                if not matches:
                    return None
                best_match = min(
                    matches,
                    key=lambda m: min(abs(m.start() - DEBIT_COL), abs(m.start() - CREDIT_COL)),
                )
                if DOCUMENT_USES_SIGNED_AMOUNTS:
                    return "debit" if best_match.group(0).startswith("-") else "credit"
                amount_pos = best_match.start()
                return "debit" if abs(amount_pos - DEBIT_COL) < abs(amount_pos - CREDIT_COL) else "credit"
        return None

    parsed_rows = []
    for line in data_lines:
        fields = line.split("|")
        if _row_is_well_formed(fields, header):
            row = dict(zip(header, fields))
        else:
            row = _realign_row_fields(fields, header)
            if row is None:
                print(f"ERROR: Could not make sense of model output line '{line}' (wrong shape, and no date/description to realign on).")
                raise SystemExit(1)
            print(f"Realigned malformed model output line '{line}' -> {row}")

        description = row.get("description", "").strip()
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
                "operation_date": normalize_operation_date(row.get("operation_date", "").strip(), infer_year=infer_year),
                "description": description,
                "debit_str": debit_str,
                "credit_str": credit_str,
                "debit": debit,
                "credit": credit,
                "resulting_balance": resulting_balance,
            }
        )

    return parsed_rows, starting_balance, stated_ending_balance
