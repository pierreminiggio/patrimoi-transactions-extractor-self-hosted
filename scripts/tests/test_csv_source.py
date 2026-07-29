"""Tests for scripts/csv_source.py.

Run from the `scripts/` directory (or anywhere, since this file adds
`scripts/` to sys.path itself):

    python3 -m unittest discover -s tests -v

or directly:

    python3 tests/test_csv_source.py
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv_source  # noqa: E402
import common  # noqa: E402


def write_temp_csv(content: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# The exact sample this feature was built against (a real Lydia — French
# neobank, BIC LYDIFRP2XXX — account-statement CSV export).
LYDIA_SAMPLE = (
    "Firstname Lastname,Pierre Miniggio\n"
    "Nom du compte,Compte principal\n"
    "Code client,10481506\n"
    "N° de compte,14391024\n"
    "IBAN,FR7617598000010001439102401\n"
    "BIC,LYDIFRP2XXX\n"
    "Période,Du 01/02/2026 au 28/02/2026\n"
    "Solde du compte au 01/02/2026,0.00 €\n"
    "Solde du compte au 28/02/2026,0.00 €\n"
    "Date,Libellé,Débit,Crédit,Solde\n"
    "05/02/2026,Rechargement par carte,,100.00,100.00\n"
    "05/02/2026,Virement SEPA émis vers IBAN FR7631233XXXXX51,-100.00,,0.00\n"
)

N26_SAMPLE = (
    "Booking Date,Value Date,Partner Name,Payment Reference,Account Number,"
    "Type,Amount (EUR),Original Amount,Original Currency,Exchange Rate\n"
    "2026-02-05,2026-02-05,Some Shop,Purchase,DE123,Card,-45.00,,,\n"
    "2026-02-06,2026-02-06,Employer,Salary,DE123,Transfer,2000.00,,,\n"
)

GENERIC_SAMPLE = (
    "Transaction Date,Description,Debit,Credit,Balance\n"
    "01/03/2026,Groceries,25.00,,975.00\n"
    "02/03/2026,Refund,,10.00,985.00\n"
)


class LydiaFormatTests(unittest.TestCase):
    def setUp(self):
        self.path = write_temp_csv(LYDIA_SAMPLE)
        self.addCleanup(os.remove, self.path)

    def test_matches_as_known_format(self):
        result = csv_source.try_known_format(self.path)
        self.assertIsNotNone(result)
        _, _, _, format_name = result
        self.assertEqual(format_name, "Lydia")

    def test_inspect_finds_real_header_past_the_metadata_preamble(self):
        fieldnames, row_count = csv_source.inspect(self.path)
        self.assertEqual(fieldnames, ["Date", "Libellé", "Débit", "Crédit", "Solde"])
        self.assertEqual(row_count, 2)

    def test_parses_rows_and_balances_correctly(self):
        parsed_rows, starting_balance, stated_ending_balance, _ = csv_source.try_known_format(self.path)

        self.assertEqual(starting_balance, 0.0)
        self.assertEqual(stated_ending_balance, 0.0)
        self.assertEqual(len(parsed_rows), 2)

        credit_row, debit_row = parsed_rows
        self.assertEqual(credit_row["operation_date"], "2026-02-05")
        self.assertEqual(credit_row["description"], "Rechargement par carte")
        self.assertEqual(credit_row["debit"], 0.0)
        self.assertEqual(credit_row["credit"], 100.0)
        self.assertEqual(credit_row["resulting_balance"], 100.0)

        self.assertEqual(debit_row["description"], "Virement SEPA émis vers IBAN FR7631233XXXXX51")
        # Lydia stores debits as negative numbers; we store the magnitude.
        self.assertEqual(debit_row["debit"], 100.0)
        self.assertEqual(debit_row["credit"], 0.0)
        self.assertEqual(debit_row["resulting_balance"], 0.0)

    def test_end_to_end_through_finalize_matches_stated_ending_balance(self):
        parsed_rows, starting_balance, stated_ending_balance, _ = csv_source.try_known_format(self.path)
        out_fd, out_path = tempfile.mkstemp(suffix=".json")
        os.close(out_fd)
        self.addCleanup(os.remove, out_path)

        common.finalize(parsed_rows, starting_balance, stated_ending_balance, out_path)

        import json
        with open(out_path, encoding="utf-8") as f:
            transactions = json.load(f)

        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0]["starting_balance"], "0.00")
        self.assertEqual(transactions[-1]["ending_balance"], "0.00")

    def test_semicolon_delimiter_variant_is_still_recognized(self):
        # Some European CSV exports (e.g. opened/re-saved in Excel) use ';'
        # instead of ','. The metadata lines and table should both parse.
        semicolon_sample = LYDIA_SAMPLE.replace(",", ";")
        path = write_temp_csv(semicolon_sample)
        self.addCleanup(os.remove, path)

        result = csv_source.try_known_format(path)
        self.assertIsNotNone(result)
        parsed_rows, starting_balance, stated_ending_balance, format_name = result
        self.assertEqual(format_name, "Lydia")
        self.assertEqual(starting_balance, 0.0)
        self.assertEqual(stated_ending_balance, 0.0)
        self.assertEqual(len(parsed_rows), 2)

    def test_comma_decimal_amounts_are_parsed(self):
        # A statement using French-style comma decimals instead of dots.
        sample = (
            "Nom du compte,Compte principal\n"
            "Solde du compte au 01/02/2026,1 234,56 €\n"
            "Solde du compte au 28/02/2026,1 334,56 €\n"
            "Date,Libellé,Débit,Crédit,Solde\n"
            "10/02/2026,Salaire,,100,00,1 334,56\n"
        )
        path = write_temp_csv(sample)
        self.addCleanup(os.remove, path)

        parsed_rows, starting_balance, stated_ending_balance, _ = csv_source.try_known_format(path)
        self.assertEqual(starting_balance, 1234.56)
        self.assertEqual(stated_ending_balance, 1334.56)

    def test_does_not_false_positive_on_unrelated_csv(self):
        self.assertIsNone(csv_source._find_lydia_table(GENERIC_SAMPLE))
        self.assertIsNone(csv_source._find_lydia_table(N26_SAMPLE))


class RegressionTests(unittest.TestCase):
    """Guard against the new Lydia tier breaking the pre-existing paths."""

    def test_n26_still_recognized_and_parsed(self):
        path = write_temp_csv(N26_SAMPLE)
        self.addCleanup(os.remove, path)

        result = csv_source.try_known_format(path)
        self.assertIsNotNone(result)
        parsed_rows, starting_balance, stated_ending_balance, format_name = result
        self.assertEqual(format_name, "N26")
        self.assertEqual(len(parsed_rows), 2)
        self.assertIsNone(starting_balance)
        self.assertIsNone(stated_ending_balance)

    def test_generic_csv_still_falls_through_to_heuristic_tier(self):
        path = write_temp_csv(GENERIC_SAMPLE)
        self.addCleanup(os.remove, path)

        self.assertIsNone(csv_source.try_known_format(path))

        result = csv_source.try_generic_parse(path)
        self.assertIsNotNone(result)
        parsed_rows, starting_balance, stated_ending_balance, column_desc = result
        self.assertEqual(len(parsed_rows), 2)


if __name__ == "__main__":
    unittest.main()
