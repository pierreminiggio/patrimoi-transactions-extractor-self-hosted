# patrimoi-transactions-extractor-self-hosted

Self-hosted version of `patrimoi-transactions-extractor`. Same trigger, same
input, same output artifact — but instead of calling the Gemini API, the
workflow downloads a small open-weight LLM and runs it locally on the GitHub
Actions runner. No AI API key or external inference provider required.

## How it differs from the hosted version

| | Hosted version | Self-hosted version |
|---|---|---|
| Model | Gemini 2.5 Flash (API) | Qwen2.5-7B-Instruct, 4-bit quantized (GGUF) |
| PDF handling | Native vision — reads raw PDF bytes | Text extracted first via `pdfplumber` (layout-preserving), then fed to the LLM as text |
| Running balance | Computed by the model | Computed deterministically in Python from extracted debit/credit values |
| Transaction order | Trusted from the model | Detected and normalized (oldest-first) deterministically in Python, regardless of source order |
| Dates | Converted by the model | Parsed and year-inferred deterministically in Python (numeric and month-name formats, multiple languages) |
| Debit/credit column | Determined by the model | Cross-checked deterministically against the raw text (column position or explicit sign, whichever the document uses) |
| Secrets needed | `GEMINI_API_KEY`, `PATRIMOI_TOKEN` | `PATRIMOI_TOKEN` only |
| Speed | Seconds | Several minutes per statement (CPU-only inference) |

### Statement formats this has been tested against

This started out tuned specifically for one bank's French statement format, then was generalized after testing against a second, very differently structured statement (English, block-based layout, reverse-chronological, explicit per-transaction running balance instead of summary balance lines). The extraction logic now handles both patterns generically rather than hardcoding either bank's vocabulary as a requirement:

- **Balance anchor**: either an explicit "balance as of \<date\>" statement (in whatever language/wording — French "SOLDE DEBITEUR/CREDITEUR AU", English "balance as of", etc.), *or* a running balance shown per-transaction (in which case the overall starting/ending balance is derived from the earliest/latest transaction's own reported balance instead).
- **Debit vs. credit**: either a fixed two-column layout (detected by comparing an amount's horizontal position to header words like "Débit"/"Crédit", "Debit"/"Credit", "Outgoing"/"Incoming", "Withdrawal"/"Deposit"), *or* an explicit sign on a single amount column (a statement that uses signs anywhere is assumed to use them consistently, so an unsigned amount is treated as a credit).
- **Dates**: numeric (`DD.MM`, `DD.MM.YYYY`) with year inferred from the statement's period when not printed, *or* fully-written dates with their own explicit year (`4 June 2026`, French or English month names).
- **Transaction order**: the model transcribes rows in whatever order the source document uses; Python detects the overall direction (by comparing the first and last parsed dates) and reverses the list if the document is newest-first, rather than trusting the model to reorder anything itself.

This is still not a universal parser — a statement with a genuinely novel structure (e.g. a third balance-anchor style, or a language neither French nor English) may need another round of generalization the same way this one did. The failure signal to watch for is the `WARNING: computed ending balance does not match...` message; treat that as "review this statement's format," not just "something's randomly wrong."

### Why so much of this is deterministic Python, not model output

Across testing on real statements, small local models proved reliably weak at exactly the same handful of things: multi-step arithmetic (running balances), sign conventions, calendar math (year inference), and maintaining a specific row order across a long transcription. Every one of those is now computed in Python from the model's raw transcription instead of trusted to the model's own reasoning — the model's job is narrowed to locating and transcribing table cells, which is what it's actually reliable at.

### Why text extraction instead of vision

There's no CPU-friendly local vision-language model that reads scanned
table layouts as reliably as Gemini's multimodal PDF understanding. Instead,
`pdfplumber` pulls the text layer out of the PDF first, and the model reasons
over that text. This works well for text-based bank statement PDFs (the
vast majority). It will **not** work for scanned/image-only PDFs, since no
OCR step is included — if you need that, `pytesseract` could be added as an
extra step.

## Model choice

**Qwen2.5-7B-Instruct**, quantized to `Q4_K_M` GGUF (~4.7 GB), run via
`llama-cpp-python` on CPU.

This was picked over smaller models (1.5B–3B) because financial transaction
extraction is a precision-sensitive structured-output task, and the extra
size meaningfully helps with instruction-following and reduces missed or
malformed rows. It was picked over larger models because 7B at Q4 is about
the practical ceiling for CPU inference within a reasonable job time on
standard GitHub-hosted runners (2 cores, 7 GB RAM).

If you want to trade accuracy for speed (or vice versa), swap the model by
changing `MODEL_REPO` / `MODEL_FILE` in the workflow, e.g.:
- Faster / lighter: `bartowski/Qwen2.5-3B-Instruct-GGUF`, `Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- More accurate, slower: `bartowski/Qwen2.5-14B-Instruct-GGUF`, `Qwen2.5-14B-Instruct-Q4_K_M.gguf`

Use bartowski's quantizations (rather than the official `Qwen/...-GGUF` repos)
because the official repos split larger quant files into multiple shards
(`-00001-of-00002.gguf`, etc.), which a plain single-file `curl` download
won't handle. bartowski publishes each quant as one file. Before swapping,
double-check the exact filename on the repo's "Files and versions" tab —
it's case-sensitive and 404s otherwise.

The model file is cached between runs via `actions/cache`, so only the
first run pays the download cost.

## Setup

1. Set the `PATRIMOI_TOKEN` repository secret (same token used to download
   the bank statement PDF as in the hosted version). No AI API key needed.
2. Trigger the workflow manually (`workflow_dispatch`) with the
   `patrimoi_bank_statement` input set to the PDF URL.
3. Download the `transactions` artifact once the run completes.

## Usage

```
gh workflow run extract-transactions.yml -f patrimoi_bank_statement="https://.../statement.pdf"
```

## Output

Same as the hosted version: a `transactions.json` artifact, a list of
objects with the following keys per transaction:

```json
{
  "operation_date": "2024-03-05",
  "description": "CARTE X 03/03 SUPERMARCHE",
  "debit": "42.10",
  "credit": "",
  "starting_balance": "1523.44",
  "ending_balance": "1481.34"
}
```

## Known limitations

- Scanned/image-only PDFs are not supported (no OCR).
- CPU inference is slow — expect several minutes per statement, well within
  the 6-hour GitHub Actions job limit but not suitable for near-real-time use.
- Small local models can still occasionally misread poorly formatted tables;
  spot-check output on statements with unusual layouts.
