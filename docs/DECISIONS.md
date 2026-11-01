# Decision log

Why each non-obvious choice was made while executing `docs/UPGRADE_PLAN.md`. Written autonomously
(owner away from keyboard), so every judgment call is recorded here for review. Newest at the bottom
of each phase. Format: **Decision**, then **Why**, then **Alternatives rejected / how to reverse**.

## Working rules applied throughout
- **No pushes, no PRs, no merges.** Each phase is a local branch stacked on the previous one. Why: the
  owner asked to "complete all tasks", not to publish; pushing is outward-facing. To publish later,
  push the last branch and open one PR, or each branch in order.
- **No Claude attribution anywhere** (commit trailers, PR text, files). Why: explicit owner instruction.
- **Ollama / live-LLM measurement deferred to Phase 1 (last).** Why: owner said to ignore Ollama for now.
  Anything needing a model is implemented and unit-tested with stubs, and its real-world effect is marked
  "not measured" in `BASELINE_METRICS.md` instead of guessed.
- **No new heavy dependencies by default.** Optional backends (reranker models, OCR) load lazily and the
  app works without them. Why: keep the local-first, easy-install property; avoid multi-GB downloads in CI.
- **Defaults must not change behaviour silently unless the plan says so.** Risky features ship behind a
  config flag with the old behaviour as fallback.

## Phase 0: housekeeping
- **Removed the `sys.path` hack, relied on `pyproject.toml` + `PYTHONPATH`/`pip install -e .`.** Why: plan
  item; the hack mutated import paths at runtime. Docker sets `PYTHONPATH=/app`; README says `pip install -e .`.
  Reverse: re-add the 3 lines at the top of `ui.py`.
- **Did not add a LICENSE or run `gh auth setup-git`.** Why: choosing a license is the owner's legal
  decision; `setup-git` edits global git config. Both listed as owner actions.
- **Ruff rules limited to `E,F`; `scripts/` exempt from style rules.** Why: matches the baseline lint metric
  so before/after is comparable; scripts are throwaway measurement code.

## Phase 2: LLM layer
- **`temperature` added as an explicit keyword on `generate()` (Protocol change) and test fakes updated**,
  instead of building a separate "grading provider". Why: smallest change that makes grading deterministic.
- **Gemini thinking budget is opt-in (`GEMINI_THINKING_BUDGET`), not defaulted.** Why: valid values differ
  per model (0 disables thinking only on some 2.5 models); a wrong default could break non-2.5 models.
- **Retry only transient errors (network, 408/425/429/5xx).** Why: retrying "model not found" or bad requests
  just delays the error message.
- **Default timeout 300 s.** Why: local models can legitimately take minutes; a short default would cut off
  valid generations. Tunable with `LLM_TIMEOUT`.
- **Prompt-injection defence = delimiters + neutralising closing tags + an instruction.** Why: cheap and
  testable offline. It reduces, not eliminates, injection risk; documented as such.

## Phase 3: pipeline refactor
- **Typed `Question` as a `TypedDict`, not a dataclass.** Why: questions flow as dicts into SQLite, PDF and
  Streamlit code; a dataclass would force a wide, risky change in a "no behaviour change" phase.
- **Proved equivalence by running old vs new code on 9 seeded scenarios**, not only by existing tests.
  Why: tests passed before and could hide subtle ordering changes (random call order).

## Phase 4: coverage-aware selection
- **Own small numpy k-means instead of adding scikit-learn.** Why: scikit-learn is a large dependency for
  one function; numpy is already installed transitively (added explicitly to `requirements.txt`).
- **Cosine k-means with fixed seed; clusters rotated in farthest-first order.** Why: a first version rotated
  clusters in arbitrary order and a unit test showed one batch could draw from several clusters of the same
  topic; farthest-first fixes that. Fixed seed = reproducible quizzes.
- **Default strategy is `coverage`, `random` kept as fallback via `SELECTION_STRATEGY`.** Why: plan calls
  coverage the biggest win; fallback keeps an escape hatch.
- **Falls back to random when embeddings are missing or the document has <= one batch of chunks.** Why:
  clustering cannot help there and short documents are the common case.
