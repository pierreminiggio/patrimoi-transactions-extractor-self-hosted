"""CSV handling, in two deterministic tiers before anything touches the LLM:

  1. Known exact provider formats (currently: N26) — matched by header,
     parsed with format-specific knowledge of what each column means.
  2. A generic heuristic parser for CSVs that don't match a known format —
     guesses which columns are the date/description/amount(s) from header
     names, then validates the guess by actually trying to parse a sample
     of rows. Only used if that validation is confident.

If neither tier can confidently parse the file, the caller falls back to
the LLM (see llm_extract.py), same as for PDFs.
"""

import csv
import re

from common import parse_number, normalize_operation_date

# Fraction of sampled data rows that must parse cleanly for the generic
# heuristic parser to trust its own column guesses.
CONFIDENCE_THRESHOLD = 0.9
SAMPLE_SIZE = 25


def read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_rows(path: str):
    text = read_text(path)
    # Sniff the delimiter rather than assuming a comma — some exports use
    # ';' (common for European CSV exports opened in Excel).
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    rows = list(reader)
    return reader.fieldnames or [], rows


# --- Tier 1: known exact formats ---------------------------------------

def _matches_n26(fieldnames):
    normalized = {(f or "").strip().lower() for f in fieldnames}
    required = {"booking date", "partner name", "payment reference", "amount (eur)"}
    return required.issubset(normalized)


def _parse_n26(path: str):
    fieldnames, rows = _read_rows(path)
    # Build a case-insensitive lookup since exact capitalization is fragile.
    key_map = {f.strip().lower(): f for f in fieldnames}

    def get(row, name):
        return (row.get(key_map.get(name, ""), "") or "").strip()

    parsed_rows = []
    for row in rows:
        booking_date = get(row, "booking date")
        if not booking_date:
            continue

        partner_name = get(row, "partner name")
        payment_reference = get(row, "payment reference")
        txn_type = get(row, "type")
        original_amount = get(row, "original amount")
        original_currency = get(row, "original currency")

        parts = [p for p in (partner_name, payment_reference) if p]
        if not parts:
            parts = [txn_type or "Transaction"]
        description = " - ".join(parts)
        if original_currency and original_currency.upper() != "EUR" and original_amount:
            description += f" ({original_amount} {original_currency})"

        amount_str = get(row, "amount (eur)")
        try:
            amount = parse_number(amount_str)
        except ValueError:
            print(f"WARNING: could not parse N26 amount '{amount_str}' on row with date {booking_date}; skipping row.")
            continue
        if amount is None:
            continue

        if amount >= 0:
            debit_str, credit_str = "", amount_str
            debit, credit = 0.0, amount
        else:
            debit_str, credit_str = amount_str, ""
            debit, credit = -amount, 0.0

        parsed_rows.append(
            {
                "operation_date": normalize_operation_date(booking_date),
                "description": description,
                "debit_str": debit_str,
                "credit_str": credit_str,
                "debit": debit,
                "credit": credit,
                "resulting_balance": None,
            }
        )

    # N26 exports don't include a running or overall balance anywhere.
    return parsed_rows, None, None


KNOWN_FORMATS = [
    ("N26", _matches_n26, _parse_n26),
]


def try_known_format(path: str):
    """Returns (parsed_rows, starting_balance, stated_ending_balance, format_name)
    or None if no known format matches this file's header.
    """
    fieldnames, _ = _read_rows(path)
    if not fieldnames:
        return None
    for name, matcher, parser in KNOWN_FORMATS:
        if matcher(fieldnames):
            parsed_rows, starting_balance, stated_ending_balance = parser(path)
            return parsed_rows, starting_balance, stated_ending_balance, name
    return None


# --- Tier 2: generic heuristic column guessing --------------------------

DATE_HEADER_RE = re.compile(r"date", re.IGNORECASE)
DATE_AVOID_RE = re.compile(r"value|due|valeur", re.IGNORECASE)

DESC_HEADER_RE = re.compile(
    r"description|label|libell|reference|r[ée]f[ée]rence|memo|narrative|details|payee|merchant|partner|counterparty|name|purpose|motif",
    re.IGNORECASE,
)
DESC_AVOID_RE = re.compile(r"iban|account\s*number|bic|swift", re.IGNORECASE)

BALANCE_HEADER_RE = re.compile(r"balance|solde", re.IGNORECASE)

AMOUNT_SINGLE_RE = re.compile(r"^amount|montant|^value$", re.IGNORECASE)
AMOUNT_AVOID_RE = re.compile(r"original|rate|exchange|taux", re.IGNORECASE)

DEBIT_CREDIT_PAIRS = [
    (r"^d[ée]bit$", r"^cr[ée]dit$"),
    (r"withdrawal", r"deposit"),
    (r"money\s*out|out", r"money\s*in|in"),
    (r"expense", r"income"),
    (r"paid\s*out", r"paid\s*in"),
]


def _guess_columns(fieldnames):
    """Return a dict describing the best-guess column mapping, or None if
    no plausible mapping was found at all (not yet validated against data).
    """
    names = [f or "" for f in fieldnames]

    # Date column: prefer one that mentions "date" but not "value"/"due".
    date_candidates = [f for f in names if DATE_HEADER_RE.search(f)]
    preferred = [f for f in date_candidates if not DATE_AVOID_RE.search(f)]
    date_col = (preferred or date_candidates or [None])[0]
    if not date_col:
        return None

    # Description columns: gather all plausible ones, up to 2, excluding
    # IBAN/account-number-like columns.
    desc_candidates = [
        f for f in names
        if DESC_HEADER_RE.search(f) and not DESC_AVOID_RE.search(f) and f != date_col
    ]
    desc_cols = desc_candidates[:2]

    # Amount: try split debit/credit columns first (higher-confidence signal).
    debit_col, credit_col = None, None
    for debit_pat, credit_pat in DEBIT_CREDIT_PAIRS:
        d = next((f for f in names if re.search(debit_pat, f, re.IGNORECASE)), None)
        c = next((f for f in names if re.search(credit_pat, f, re.IGNORECASE)), None)
        if d and c and d != c:
            debit_col, credit_col = d, c
            break

    amount_col = None
    if not debit_col:
        amount_candidates = [
            f for f in names
            if AMOUNT_SINGLE_RE.search(f) and not AMOUNT_AVOID_RE.search(f)
        ]
        if len(amount_candidates) == 1:
            amount_col = amount_candidates[0]
        elif len(amount_candidates) > 1:
            exact = [f for f in amount_candidates if f.strip().lower() == "amount"]
            amount_col = exact[0] if exact else None

    if not debit_col and not amount_col:
        return None

    balance_col = next((f for f in names if BALANCE_HEADER_RE.search(f)), None)

    return {
        "date_col": date_col,
        "desc_cols": desc_cols,
        "debit_col": debit_col,
        "credit_col": credit_col,
        "amount_col": amount_col,
        "balance_col": balance_col,
    }


def _try_parse_date_generic(raw):
    """Lightweight validation-only date check: does this look like a date
    we could normalize? Doesn't require a year-inference resolver since
    the vast majority of CSV exports include a full date with year.
    """
    raw = (raw or "").strip()
    if not raw:
        return False
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return True
    if re.match(r"^\d{1,2}[./-]\d{1,2}[./-]\d{4}$", raw):
        return True
    if re.match(r"^\d{1,2}\s+[A-Za-zÀ-ÿ]+\s+\d{4}$", raw):
        return True
    return False


def _validate_mapping(mapping, rows):
    sample = rows[:SAMPLE_SIZE] if len(rows) > SAMPLE_SIZE else rows
    if not sample:
        return False

    date_ok = sum(1 for r in sample if _try_parse_date_generic(r.get(mapping["date_col"], "")))
    if date_ok / len(sample) < CONFIDENCE_THRESHOLD:
        return False

    if mapping["debit_col"]:
        amount_ok = 0
        for r in sample:
            d = (r.get(mapping["debit_col"], "") or "").strip()
            c = (r.get(mapping["credit_col"], "") or "").strip()
            try:
                if (d and parse_number(d) is not None) or (c and parse_number(c) is not None):
                    amount_ok += 1
            except ValueError:
                pass
    else:
        amount_ok = 0
        for r in sample:
            v = (r.get(mapping["amount_col"], "") or "").strip()
            try:
                if v and parse_number(v) is not None:
                    amount_ok += 1
            except ValueError:
                pass

    return amount_ok / len(sample) >= CONFIDENCE_THRESHOLD


def try_generic_parse(path: str):
    """Returns (parsed_rows, starting_balance, stated_ending_balance, description)
    or None if the heuristic couldn't confidently map columns.
    """
    fieldnames, rows = _read_rows(path)
    if not fieldnames or not rows:
        return None

    mapping = _guess_columns(fieldnames)
    if mapping is None:
        return None
    if not _validate_mapping(mapping, rows):
        return None

    parsed_rows = []
    for row in rows:
        date_raw = (row.get(mapping["date_col"], "") or "").strip()
        if not date_raw:
            continue

        desc_parts = [
            (row.get(c, "") or "").strip()
            for c in mapping["desc_cols"]
            if (row.get(c, "") or "").strip()
        ]
        description = " - ".join(desc_parts) if desc_parts else "Transaction"

        try:
            if mapping["debit_col"]:
                debit_str = (row.get(mapping["debit_col"], "") or "").strip()
                credit_str = (row.get(mapping["credit_col"], "") or "").strip()
                debit = parse_number(debit_str) or 0.0
                credit = parse_number(credit_str) or 0.0
            else:
                amount_str = (row.get(mapping["amount_col"], "") or "").strip()
                amount = parse_number(amount_str)
                if amount is None:
                    continue
                if amount >= 0:
                    debit_str, credit_str = "", amount_str
                    debit, credit = 0.0, amount
                else:
                    debit_str, credit_str = amount_str, ""
                    debit, credit = -amount, 0.0
        except ValueError:
            continue

        resulting_balance = None
        if mapping["balance_col"]:
            try:
                resulting_balance = parse_number((row.get(mapping["balance_col"], "") or "").strip())
            except ValueError:
                resulting_balance = None

        parsed_rows.append(
            {
                "operation_date": normalize_operation_date(date_raw),
                "description": description,
                "debit_str": debit_str,
                "credit_str": credit_str,
                "debit": debit,
                "credit": credit,
                "resulting_balance": resulting_balance,
            }
        )

    if not parsed_rows:
        return None

    if mapping["debit_col"]:
        column_desc = f"date='{mapping['date_col']}', debit='{mapping['debit_col']}', credit='{mapping['credit_col']}'"
    else:
        column_desc = f"date='{mapping['date_col']}', amount='{mapping['amount_col']}'"
    if mapping["desc_cols"]:
        column_desc += f", description={mapping['desc_cols']}"
    if mapping["balance_col"]:
        column_desc += f", balance='{mapping['balance_col']}'"

    return parsed_rows, None, None, column_desc
