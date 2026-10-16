# AGENTS.md

Guidelines for any AI agent (or human) changing this project.

## 1. Development method: TDD is mandatory

Every bug fix and every feature follows test-driven development:

1. **Red** — write a failing test first that reproduces the bug or specifies the
   new behavior. Run it and confirm it fails for the expected reason (for a
   bug, with the real error message).
2. **Green** — write the smallest change that makes the test pass.
3. **Refactor / regress** — run the full suite and make sure nothing else broke.

Do not change production code without a test that failed before the change.
Tests live in `scripts/tests/`. Run them from the project root with the bash
script (Git Bash on Windows, or macOS/Linux):

    ./run-tests

or with the equivalent cross-platform command:

    python -m unittest discover -s scripts/tests -v

Note: `scripts/run-tests` is a one-line shell snippet, not a Python script, so
`python scripts/run-tests` does not work.

Expected noise: some tests deliberately print log lines (e.g. "Realigned
malformed model output line", "ERROR: Could not make sense of ..."). That is
normal as long as the run ends with `OK`.

Tests must not need the model, the network or real bank data. Use sanitized
statement text and hand-written model outputs (see
`scripts/tests/test_llm_extract.py`). Never commit real account data.

## 2. How to deliver changes

- Deliver the changes as a **zip file** containing only the files that were
  edited or created, at their **correct paths relative to the project root**
  (no extra wrapper folder), so it can be dragged and dropped over the project.
  `AGENTS.md` itself goes at the root of the zip if it was created or changed.
- Never include `__pycache__`, caches, the model file or other build artifacts.
- If files must be deleted, do **not** put them in the zip. Print, in the chat,
  the list of `rm` commands (paths relative to the project root) to run to
  delete them cleanly. If nothing has to be deleted, say so explicitly.
