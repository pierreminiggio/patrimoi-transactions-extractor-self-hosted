"""Tests for scripts/llm_extract.py (parsing of the model's raw output).

Run from the `scripts/` directory:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import llm_extract  # noqa: E402

HEADER = "operation_date|description|debit|credit|resulting_balance"

# Sanitized layout of a BNP Paribas "RELEVE DE COMPTE CHEQUES" statement.
SOURCE_TEXT = """\
                     RELEVE DE COMPTE CHEQUES
               du 13 septembre 2026 au 13 octobre 2026

 Date   Nature des operations                        Valeur            Debit        Credit
        SOLDE CREDITEUR AU 13.09.2026                                                1,80
 25.09  VIR SEPA INSTANT RECU /DE M PIERRE           25.09                           0,81
        MINIGGIO /REF NOTPROVIDED /MOTIF FRAIS
 02.10  *FRAIS TENUE DE COMPTE                       02.10             2,60
 09.10  VIR SEPA INSTANT RECU /DE M PIERRE           09.10                         250,00
        MINIGGIO /REF NOTPROVIDED /MOTIF EPARGNE BNP
 09.10  VIR CPTE A CPTE EMIS /MOTIF EPARGNE240       16.10           240,00
        /BEN MINIGGIO PIERRE CLAUDE RAYMOND
        TOTAL DES OPERATIONS                                         242,60        250,81
        SOLDE CREDITEUR AU 13.10.2026                                               10,01
"""


def model_output(*rows, start="1.80", end="10.01"):
    return "\n".join([f"STARTING_BALANCE: {start}", f"ENDING_BALANCE: {end}", HEADER, *rows])


class ParseLlmOutputTests(unittest.TestCase):
    def test_well_formed_rows_are_parsed(self):
        out = model_output(
            "25.09|VIR SEPA INSTANT RECU /DE M PIERRE MINIGGIO /REF NOTPROVIDED /MOTIF FRAIS||0.81|",
            "02.10|*FRAIS TENUE DE COMPTE|2.60||",
            "09.10|VIR SEPA INSTANT RECU /DE M PIERRE MINIGGIO /REF NOTPROVIDED /MOTIF EPARGNE BNP||250.00|",
            "09.10|VIR CPTE A CPTE EMIS /MOTIF EPARGNE240 /BEN MINIGGIO PIERRE CLAUDE RAYMOND|240.00||",
        )
        rows, start, end = llm_extract.parse_llm_output(out, SOURCE_TEXT)
        self.assertEqual(start, 1.80)
        self.assertEqual(end, 10.01)
        self.assertEqual([(r["debit"], r["credit"]) for r in rows],
                         [(0.0, 0.81), (2.60, 0.0), (0.0, 250.0), (240.0, 0.0)])
        self.assertEqual(rows[0]["operation_date"], "2026-09-25")

    def test_row_with_stray_leading_amount_and_shifted_fields_is_repaired(self):
        # Real failure: the model emitted a spurious leading number, which
        # shifted every field one slot to the right, so the description
        # landed in the `debit` column and float() blew up.
        out = model_output(
            "442.60|24.08|VIR CPTE A CPTE EMIS /MOTIF EPARGNE /BEN 01.09|440.00||",
        )
        rows, _, _ = llm_extract.parse_llm_output(out, SOURCE_TEXT)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["description"], "VIR CPTE A CPTE EMIS /MOTIF EPARGNE /BEN")
        self.assertEqual(row["debit"], 440.0)
        self.assertEqual(row["credit"], 0.0)
        self.assertEqual(row["debit_str"], "440.00")
        self.assertEqual(row["credit_str"], "")
        self.assertTrue(row["operation_date"].endswith("-08-24"), row["operation_date"])
        # The stray leading value is not a trustworthy balance.
        self.assertIsNone(row["resulting_balance"])

    def test_repair_does_not_disturb_neighbouring_rows(self):
        out = model_output(
            "02.10|*FRAIS TENUE DE COMPTE|2.60||",
            "442.60|09.10|VIR CPTE A CPTE EMIS /MOTIF EPARGNE240|240.00||",
            "09.10|VIR SEPA INSTANT RECU /DE M PIERRE MINIGGIO||250.00|",
        )
        rows, _, _ = llm_extract.parse_llm_output(out, SOURCE_TEXT)
        self.assertEqual([r["debit"] for r in rows], [2.60, 240.0, 0.0])
        self.assertEqual([r["credit"] for r in rows], [0.0, 0.0, 250.0])

    def test_unrecoverable_row_still_fails_loudly(self):
        # No description / date to anchor a repair on: silently guessing
        # would corrupt the data, so the run must still stop.
        out = model_output("garbage|||||")
        with self.assertRaises(SystemExit):
            llm_extract.parse_llm_output(out, SOURCE_TEXT)


if __name__ == "__main__":
    unittest.main()
