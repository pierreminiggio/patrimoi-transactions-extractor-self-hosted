"""Shared helpers used by every extraction path (PDF+LLM, deterministic CSV,
CSV-fallback+LLM). Keeping these in one place means "what counts as a valid
date", "how running balance is computed", etc. behave identically no matter
which path produced the raw rows.
"""

import json
import re

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


def parse_number(raw: str):
    """Parse a number that may use either '.' or ',' as the decimal
    separator, and may include thousands separators (spaces, apostrophes,
    or the other punctuation mark). Returns None for an empty string,
    raises ValueError for genuinely unparseable input.
    """
    if raw is None:
        return None
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


def normalize_operation_date(raw: str, infer_year=None) -> str:
    """Normalize a date string to YYYY-MM-DD.

    infer_year, if given, is a callable(month:int) -> int | None used to
    resolve day/month-only dates (e.g. "09.05") that have no year of their
    own. Callers that already have a statement period (PDF path) pass a
    resolver; callers whose dates already carry a year (most CSV exports)
    can omit it.
    """
    raw = (raw or "").strip()
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
    # DD.MM or DD/MM — no year; infer it if we have a resolver.
    m = re.match(r"^(\d{1,2})[./-](\d{1,2})$", raw)
    if m:
        day, month = m.groups()
        year = infer_year(int(month)) if infer_year else None
        if year is None:
            print(f"WARNING: could not infer a year for date '{raw}'; leaving it un-converted.")
            return raw
        return f"{year:04d}-{int(month):02d}-{int(day):02d}"
    print(f"WARNING: unrecognized date format '{raw}'; leaving it un-converted.")
    return raw


def _looks_ascending(rows):
    iso_dates = [r["operation_date"] for r in rows if re.match(r"^\d{4}-\d{2}-\d{2}$", r["operation_date"])]
    if len(iso_dates) < 2:
        return True  # not enough evidence to tell; assume already correct
    return iso_dates[0] <= iso_dates[-1]


def finalize(parsed_rows, starting_balance, stated_ending_balance, output_path):
    """Shared tail end of every extraction path:
    - normalize row order to oldest-first (reversing if the source was newest-first)
    - derive starting/ending balance from a per-row resulting_balance anchor
      if no explicit balance was found
    - compute the running balance deterministically
    - write transactions.json
    - cross-check the computed ending balance against the stated one
    """
    if not _looks_ascending(parsed_rows):
        parsed_rows.reverse()

    if starting_balance is None:
        anchor = next((r for r in parsed_rows if r.get("resulting_balance") is not None), None)
        if anchor is not None:
            starting_balance = anchor["resulting_balance"] - anchor["credit"] + anchor["debit"]
            print(
                f"No explicit starting balance found; derived "
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
        anchor = next((r for r in reversed(parsed_rows) if r.get("resulting_balance") is not None), None)
        if anchor is not None:
            stated_ending_balance = anchor["resulting_balance"]
            print(
                f"No explicit ending balance found; using the most recent "
                f"transaction's own reported balance ({stated_ending_balance:.2f}) "
                f"instead."
            )

    transactions = []
    running_balance = starting_balance
    for r in parsed_rows:
        row_starting_balance = running_balance
        running_balance = running_balance - r["debit"] + r["credit"]
        transactions.append(
            {
                "operation_date": r["operation_date"],
                "description": r["description"],
                "debit": f"{r['debit']:.2f}" if r.get("debit_str") else "",
                "credit": f"{r['credit']:.2f}" if r.get("credit_str") else "",
                "starting_balance": f"{row_starting_balance:.2f}",
                "ending_balance": f"{running_balance:.2f}",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transactions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(transactions)} transactions to {output_path}")
    print(f"Computed ending balance: {running_balance:.2f}")

    if stated_ending_balance is not None:
        print(f"Statement's stated/derived ending balance: {stated_ending_balance:.2f}")
        if abs(running_balance - stated_ending_balance) > 0.01:
            print(
                "WARNING: computed ending balance does not match the statement's "
                "stated ending balance. This usually means a transaction was "
                "missed, misread, or a debit/credit was misclassified — review "
                f"{output_path} against the original source before trusting it."
            )
    else:
        print("No ending balance could be determined from the source to validate against.")
