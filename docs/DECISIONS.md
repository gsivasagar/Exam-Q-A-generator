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

## Phase 5: embedder upgrade
- **Built the mechanism but kept `default` (MiniLM) as the default model.** Why: the plan says choose by
  recall@k on a labelled set, which needs Ollama and the Phase 1 harness. Switching the default without
  a measurement would be a guess. After Phase 1 runs, change `EMBEDDING_MODEL`'s default in `config.py`.
- **`ollama:<model>` spec string instead of separate provider/model settings.** Why: one env var, easy to
  extend (`gemini:...`) later, and the collection name derives from it.
- **One Chroma collection per embedder; default keeps the old name `exam_chunks`.** Why: vectors from
  different models have different sizes, so mixing would crash or silently corrupt retrieval; keeping
  the old name means existing stores need no migration.
- **Own `OllamaEmbedder` (using the `ollama` client) rather than Chroma's built-in Ollama function.**
  Why: reuses our timeout/host config, gives friendly `EmbeddingError` (a subclass of `LLMError`, so the
  UI's existing error handling shows "is Ollama running / ollama pull X"), and is testable with a fake client.
- **Replaced `lru_cache` on `_collection()` with a cache keyed by (path, embedder).** Why: the old
  zero-argument cache silently reused a stale store if the directory or embedder changed (issue #3 from
  the review). `cache_clear` is re-exported so existing test fixtures keep working.

## Phase 6: chunking
- **Shipped behind `CHUNKING=structured`; default remains `legacy`.** Why: the plan's gate is "retrieval metrics
  up or flat, citation correctness up". Neither is measurable without the Phase 1 harness and a model, so
  switching the default now would be unverified. Flip it in `config.py` after Phase 1 shows it helps.
- **Heading detection is heuristic** (numbered, "Chapter N", ALL CAPS, short Title Case). Why: PDF text has
  no real structure and a layout-aware parser would add a heavy dependency. Known weakness: documents with
  many Title Case lines (e.g. resumes) look like they have many headings.
- **A heading only forces a new chunk once the current chunk reaches half the target size.** Why: the first
  version split at every heading and produced 377 chunks with a median of 62 tokens on the sample PDFs
  (one resume-like PDF alone gave 339 tiny chunks). Folding short sections together gave 163 chunks with a
  floor of 82 tokens. Trade-off: a chunk is labelled with the heading it *starts* under and can contain a
  following short section.
- **Reported honestly that the median chunk (218 tokens) is not closer to the 400-token target than legacy
  (282).** The improvement is the tighter range and no fragments, and that chunks now carry headings and can
  cross pages.
- **Parent expansion is off by default and applied only to topic searches.** Why: expanded parents are ~3x
  larger and would crowd the 6,000-character prompt cap; the no-topic path already uses every chunk.
- **Found and fixed a bug via tests:** `"1 2 3"` was classified as a Title Case heading (vacuous `all()`).

## Phase 7: hybrid retrieval and reranking
- **Implemented BM25 myself (~50 lines) instead of adding `rank_bm25`.** Why: doc sets are small, the
  algorithm is simple and fully unit-testable, and it avoids a dependency.
- **BM25 index is built per query from the document set's chunks, not persisted.** Why: sets are per quiz and
  small (tens to hundreds of chunks); persisting would add invalidation bugs for no measurable gain.
- **Reciprocal Rank Fusion (k=60) rather than score blending.** Why: BM25 and cosine scores are on different
  scales; rank-based fusion needs no tuning.
- **Reranker is an optional backend (`fastembed:<model>`), lazily imported, never a hard dependency.** Why:
  cross-encoders need a model download (hundreds of MB) and an extra package; the app must stay installable
  and testable offline. **Caveat:** the real fastembed path is unit-tested only with a fake encoder, because
  no model was downloaded in this session. Verify it with a real model before relying on it.
- **BGE reranker suggested as the first model to try; Qwen reranker not wired in.** Why: sources disagree on
  Qwen3-Reranker's license (plan section 6), so the permissively licensed option is the safer default to try.
- **Defaults unchanged (`dense`, no reranker, no distance cutoff).** Why: no evidence yet. The Phase 1
  retrieval evaluation needs only embeddings (not an LLM), so it can run offline and decide.
- **"Topic not found" only triggers when `RETRIEVAL_MAX_DISTANCE` is set.** Why: a sensible threshold depends
  on the embedder and must be calibrated on data; guessing one could wrongly reject valid topics.

## Phase 8: adaptive router
- **Routing is on by default (`ADAPTIVE_ROUTING=on`) but `FULL_CONTEXT_BATCH` defaults to 5 (no change).**
  Why: sending the whole document is low-risk and fixes real omissions (a 5-chunk document lost 1 chunk per
  call). Asking a small local model for 10 questions per call is a quality risk I cannot measure offline, so
  that stays opt-in. Reverse routing with `ADAPTIVE_ROUTING=off`.
- **Budget is per provider, not global.** Ollama's default window is small, so the default stays at the old
  6,000-character cap unless `OLLAMA_NUM_CTX` is set; Gemini declares 120,000 characters (its window is far
  larger, so this only bounds cost/latency). A fake or unknown provider falls back to the config default.
- **Topic quizzes on short documents run retrieval first as a relevance check, then use the full text.**
  Why: Phase 7 promised that a topic absent from the notes fails gracefully; routing straight to "full"
  would have silently dropped that behaviour (my first test caught the conflict).
- **Added an optional `on_route` callback and a one-line UI caption instead of a return-type change.** Why:
  `generate_qa_pairs` returns a list that tests, UI and exports rely on; a callback is backward compatible.
- **Honest limit:** the router decides on characters, not tokens (about 4 chars/token); fine for sizing, but
  not exact for CJK or code-heavy text.

## Phase 9: question quality
- **Verifier is off by default and fails open.** Why: it adds an LLM call per batch (cost/latency unmeasured),
  and a flaky verifier must never empty a quiz. If its reply is unusable or the call fails, all questions
  are kept. Questions with no cited passage are kept too (nothing to check them against).
- **Verifier runs per batch inside `_generate_batch`, where the cited passage text is available.** Why:
  verifying after the fact would need to carry passage text through the whole pipeline. The citation index
  travels as an internal `_chunk` key and is stripped before results are returned, so no downstream code
  (DB, UI, PDF export) sees it. `validate_pairs` only keeps it when asked, so existing callers are unchanged.
- **True/false is its own type `tf` that reuses the multiple-choice UI and grading** (two options, graded
  deterministically, no LLM call). Why: cheapest to build, no new grading path, and offline-testable.
- **Cloze is graded by the LLM like short answers.** Why: exact string match would fail on synonyms or
  inflection ("mitochondria" vs "mitochondrion"). Cost: one grading call per cloze question.
- **`explanation` is optional in the schema (not in `required`).** Why: weaker local models may omit it; a
  required field would make their whole reply invalid and lower the valid-question rate.
- **Semantic dedupe stays off; default threshold lowered from 0.9 to 0.85.** Why: a quick test with the real
  embedder showed 0.9 catches only 3 of 8 paraphrases while 0.85 catches 7 of 8, with the same single false
  drop (templated questions such as "Who wrote Hamlet?" vs "Who wrote Macbeth?" score ~0.9). With only 16
  pairs this is a hint, not proof, so it is left to the Phase 1 evaluation before enabling.
- **`mixed` still means short + multiple-choice only.** Why: changing it would alter existing behaviour and
  saved expectations; the new types are selected explicitly.
