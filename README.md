# patrimoi-transactions-extractor-self-hosted

Self-hosted version of `patrimoi-transactions-extractor`. Same trigger, same
input, same output artifact — but instead of calling the Gemini API, the
workflow downloads a small open-weight LLM and runs it locally on the GitHub
Actions runner. No AI API key or external inference provider required.

## How it differs from the hosted version

| | Hosted version | Self-hosted version |
|---|---|---|
| Model | Gemini 2.5 Flash (API) | Qwen2.5-7B-Instruct, 4-bit quantized (GGUF) |
| PDF handling | Native vision — reads raw PDF bytes | Text extracted first via `pdfplumber`, then fed to the LLM as text |
| Running balance | Computed by the model | Computed deterministically in Python from the extracted debit/credit values and the model-extracted starting balance |
| Secrets needed | `GEMINI_API_KEY`, `PATRIMOI_TOKEN` | `PATRIMOI_TOKEN` only |
| Speed | Seconds | Several minutes per statement (CPU-only inference) |

### Why the running balance is computed in Python, not by the model

Gemini's larger scale makes it reasonably reliable at carrying a running sum
across many rows. A 7B model on CPU is more prone to arithmetic drift over
a long transaction list. So the local model is only asked to extract the
raw fields it's actually good at (dates, descriptions, debit/credit amounts,
and the statement's single starting balance) — the sequential balance math
is then done deterministically in the script. This produces the same
`transactions.json` shape as the hosted version, just computed more
reliably.

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
- Faster / lighter: `Qwen/Qwen2.5-3B-Instruct-GGUF`, `qwen2.5-3b-instruct-q4_k_m.gguf`
- More accurate, slower: `Qwen/Qwen2.5-14B-Instruct-GGUF`, `qwen2.5-14b-instruct-q4_k_m.gguf`

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
