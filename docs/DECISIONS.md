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

## Phase 10: concurrency, caching, progress
- **Default concurrency is 1 for Ollama and 4 for Gemini.** Why: one local model usually serialises requests
  anyway (and parallel requests can raise memory use); the cloud API is where parallelism pays off. Users with
  a capable GPU can set `OLLAMA_CONCURRENCY` equal to Ollama's `OLLAMA_NUM_PARALLEL`.
- **Generation parallelism uses "waves" planned up front (`plan_wave`), not free-running threads.** Why: which
  batch to request next depends on the results so far (quota, failed types). A wave assumes each request fills
  its share of the shortfall, then re-plans after the wave. With one worker this reduces to exactly the old
  loop; I verified the output is identical to the previous commit on 9 seeded scenarios.
- **Chunk selection stays single-threaded; only the LLM calls run in the pool.** Why: the cyclers use shared
  state and `random`; keeping them in the caller thread keeps quizzes reproducible given a seed (for 1 worker).
- **Progress callbacks are always invoked from the calling thread.** Why: Streamlit UI objects must not be
  touched from worker threads.
- **The first worker error is re-raised only after the pool drains.** Why: avoids abandoned in-flight calls
  and matches the sequential behaviour of failing with the original error.
- **Response cache is opt-in (`LLM_CACHE_DIR`) and never caches failures.** Why: with it on, identical prompts
  give identical quizzes, which users of the app would not expect; it is meant for development and the
  evaluation harness (repeatable, free re-runs).
- **No `st.cache_resource` change.** The plan asked for it, but `_open()` is already `lru_cache`d per
  (path, embedder) at process level, which survives Streamlit reruns, so a second cache layer would add nothing.
- **Latency numbers are simulated.** I state this in the metrics log; they show the mechanism works, not what
  a real model will deliver.

## Phase 11: grading
- **Rubric scoring: the model only says which key points are covered; the score is computed in code.** Why:
  models are poor at calibrated numbers but decent at yes/no per point. Scores become explainable (a checklist)
  and repeatable by construction (discrete values), which is the plan's goal.
- **Rubric applies to short questions only, and only when 2+ key points exist; otherwise the old holistic
  grader runs.** Why: weaker local models may omit `key_points` (the field is optional in the schema), and
  cloze answers are single terms. Fallback means no question is ever ungradable because of this feature.
- **A malformed rubric reply becomes "ungraded" (score None) rather than silently falling back to holistic.**
  Why: a fallback would double the cost on a flaky model and hide the failure; the new "Retry ungraded"
  button handles recovery.
- **Pre-checks are conservative.** Blank and a short list of "no attempt" phrases score 0; a normalised exact
  match or >= 95% similarity scores 1. Anything else goes to the model. Why: a wrong shortcut would mislead
  students; the 0.95 threshold was chosen to forgive punctuation/case only. Trade-off: a student who copies the
  reference answer gets full marks (same as before, but no longer costs a call).
- **Exact-match test updated:** an existing test expected one LLM call for an answer identical to the reference;
  the pre-check now (intentionally) skips the call, so the test uses a near-miss answer instead.
- **Retry stores only the newly graded answers.** Why: the originals were never saved (ungradable answers are
  skipped by `store_results`), so storing the recovered ones avoids duplicates.

## Phase 12: ingestion
- **DOCX/PPTX read with `zipfile` + `ElementTree`, not python-docx / python-pptx.** Why: both formats are zip
  files of XML and we only need text; avoids two dependencies. Trade-off: no tables-as-structure, no
  headers/footers, no embedded images. Zip members are size-bounded to refuse zip bombs.
- **Format is detected from content (magic bytes), the extension is only a fallback for text files.** Why: a
  renamed or mislabelled upload should not bypass validation or be mis-parsed.
- **Pages for DOCX/Markdown/text/web are 0, meaning "no page", not 1.** Why: citing "p.1" for a Word document
  would be false. The UI already shows a page only when it is truthy. PPTX uses the slide number as the page.
- **URL ingestion refuses private / loopback / link-local / reserved addresses and does not follow redirects.**
  Why: server-side fetching of user-supplied URLs is a classic SSRF hole (reaching localhost services or cloud
  metadata). A redirect could point back at an internal address, so the user must paste the final address.
  Known residual risk: DNS rebinding between the check and the fetch is not fully closed (the resolver result
  is not pinned for the connection). For a single-user local app this is acceptable; pin the IP before
  exposing the app publicly.
- **OCR is an optional backend, off unless the packages and Tesseract are present.** Why: Tesseract is a
  system program (not pip-installable) and models are large. The code path is tested only with an injected
  fake recogniser; verify with a real scanned PDF before relying on it.
- **Kept `pypdf` as the extractor even though `pypdfium2` is ~12x faster with equal output.** Why: extraction
  speed is negligible next to LLM calls, switching changes every chunk (and so retrieval results) without an
  evaluation to justify it, and pypdfium2 is a compiled dependency. Revisit after Phase 1. pdfplumber was
  rejected (slower, many glued words).
- **Found a real bug while testing:** UTF-16 "decodes" almost any byte string, so Latin-1 text came out as
  garbage. UTF-16 is now tried only when a byte-order mark is present.
- **Added `fonttools` to requirements.** Why: pypdf warns it is needed to fully decode some font encodings; a
  clean Docker install would not have it. Output on the sample PDFs was identical either way.

## Phase 13: persistence and product features
- **No real accounts; an optional "profile name" instead.** Why: the plan said to defer accounts. Real
  authentication (passwords, sessions, reset flows) is large and easy to get wrong. A profile name gives
  history that survives a refresh at no security cost, but it is **not private**: anyone typing the same name
  sees that history. It is labelled as a label, not a login, in the UI and README. Use real auth before
  exposing the app to other people.
- **Migrations tracked with SQLite's `PRAGMA user_version`, written as numbered steps.** Why: no extra
  library, works for existing databases (a pre-versioning DB is treated as version 0 and upgraded in place;
  tested with data), and each future schema change is one new `if version < N` block.
- **Questions are stored as JSON blobs in `quizzes` / `srs`.** Why: question shape keeps evolving (key points,
  explanations, new types); JSON avoids a migration per field. Trade-off: no SQL queries over question fields.
- **Retaking reuses the quiz id, with a fresh `attempt_id` for the form widgets.** Why: results from every
  attempt accumulate under one quiz (attempt count and average score), while new widget keys stop the old
  answers from reappearing in the form.
- **Spaced repetition is a simplified SM-2** (reps 1 and 2 -> 1 and 3 days, then interval x ease; a fail resets
  to 1 day and lowers ease; ease bounded 1.3-2.8). Why: well known and tiny; graded by the existing 0.6 pass
  mark. Cards are identified by a hash of question + answer, so regenerated identical questions share a card.
- **"Review due" needs no LLM to build the quiz**: due questions are stored with their options and answers.
  Weak-topic practice is hidden when there is no live document set (e.g. after a review or retake) because it
  needs the notes.
- **Exports are plain stdlib/fpdf**, with Anki text HTML-escaped and tabs/newlines flattened so imports do not
  break; CSV uses a BOM-prefixed UTF-8 download so Excel reads accents correctly.
- **Document cleanup runs once per browser session at startup and swallows errors.** Why: no background
  scheduler to manage; a broken store must never stop the app from opening. Documents are registered when
  created and forgotten when explicitly deleted.
- **Moved the history sidebar blocks to the end of the script.** Streamlit draws top to bottom, so a list
  built before the quiz is created is stale until the next click (caught by a failing test).

## Phase 14: DevOps and hardening
- **Actually built and ran the image instead of only writing it.** Docker Desktop was installed but stopped; I
  started it. This found a real problem (below) that reading the Dockerfile would not have. I removed my test
  images afterwards and left the pre-existing `exam-qa-generator-app` image alone.
- **Baked the embedding model (~80 MB) into the image.** Why: without it, the first upload in a fresh container
  waited 120 s for a download (slow connection here) and failed completely without internet. After baking it in,
  ingestion took 1.4 s with the network disabled. Cost: image +180 MB after deleting the leftover `.tar.gz`; it
  is still smaller than the old image.
- **Dropped `build-essential` and `curl` and ran as a non-root user.** Why: the pinned wheels install without a
  compiler (the build succeeded), curl was only used by the health check (replaced with a Python one-liner), and
  non-root limits damage if the app is compromised.
- **Pinned dependencies in `requirements.lock`, used only by Docker.** `requirements.txt` stays the human-edited
  source with loose bounds. The lock is compiled for Linux/Python 3.11. It is not used for local installs
  because this Mac (Intel) has no wheel for the newest `onnxruntime` that the lock selects.
- **Named volume for data instead of `./data`.** Why: a bind mount created by Docker on Linux is owned by root
  and the non-root container user cannot write to it; a named volume avoids that on every OS. Documented how to
  switch back.
- **mypy is blocking in CI with `ignore_missing_imports`.** Why: the 23 findings were small and genuine
  (implicit Optional defaults, an unchecked regex match, untyped containers), so fixing them was cheaper than
  carrying an exemption. Strict mode was not enabled.
- **Coverage floor 92%, with current 94.6%.** Why: a floor a couple of points below reality catches
  regressions without failing on noise.
- **Docker job runs only on pushes to main.** Why: the build takes ~5 minutes and would slow every PR.
- **The scheduled evaluation workflow is created together with the evaluation harness (Phase 1)** because it
  has nothing to run until then.
- **Did not add a LICENSE.** Choosing one is the owner's decision.

## Phase 1: evaluation harness (run last, as requested)
- **Wrote my own study notes and queries instead of using your PDFs.** Why: the PDFs in `tmp/` are personal
  documents (a CV and project write-ups) that should not be copied into the repository, and most are 2-5 chunks long,
  too small to test retrieval. The 8 authored notes cover related topics that are easy to confuse (cell energy vs
  photosynthesis, algorithms vs databases, thermodynamics vs economics) so retrieval is not trivial. Limitation:
  I wrote both the notes and the questions, so the benchmark favours well-formed, fact-like questions.
- **Scaled the chunk sizes down for the retrieval benchmark** (60 words / 90 tokens instead of 300 / 400). Why: the
  notes are short; at full size each would be one or two chunks and retrieval would be trivial. This emulates long
  documents but is a stand-in, so absolute numbers are indicative only.
- **Judged every comparison with a paired bootstrap, not by eyeballing the table.** Why: with 68 queries the 95%
  intervals are about +/-0.05 MRR; most differences I first read as wins were noise. My rule: change a default only
  if the paired interval excludes zero (and nothing important regresses).
- **Result: no default was flipped for retrieval or chunking.** Hybrid, structured chunking and parent expansion are
  not distinguishable from the current default. The reranker helps but costs ~0.5-0.75 s per topic query and needs an
  optional package, so it is documented and opt-in rather than default.
- **Kept `coverage` as the default chunk-selection strategy, after fixing it.** The first Phase 4 version was weaker
  than I claimed (see below). The source-aware version is strongly better in both measures, so the default is now
  justified by data it did not have before.
- **Added `fastembed:` as an embedder/reranker backend.** Why: it lets the retrieval pipeline be tested with real
  models without an Ollama server, which found two real bugs and validated the reranker code path. It is optional.
- **Stand-ins (`StubGenerator`, `LexicalGrader`) instead of skipping the generation/grading evaluation.** Why: they
  keep the harness runnable and tested in CI and provide a floor (the lexical grader agrees with the labels 61-64% of
  the time). They are clearly labelled as pipeline probes; nothing in the docs presents them as model quality.
- **Regression floors (`eval/thresholds.json`) sit a little below today's results** and run weekly in CI, so a real
  quality drop fails the job while noise does not. Raise them when quality improves.
- **Honest corrections** (also in BASELINE_METRICS section 9): clustering-only selection was not better on documents
  touched; the Ollama/fastembed embedders crashed on search until Phase 1; hybrid + parent expansion silently lost its
  dense half; `generate_qa_pairs` complexity had crept back to 21 and was restored to 5. Two evaluation-tool bugs of my
  own were caught by sanity-checking implausible numbers (a grader scoring everything correct; a duplicate rate
  above 100%).
- **Calibration labels are mine.** The 33 grading labels (correct/partial/wrong) were written by me. They are a
  starting point; replace or extend them with real marking before trusting absolute grading numbers.

## Phase 15: optional advanced work
- **Built nothing, on purpose.** The plan said to do these only if the evaluation shows a gap worth the cost.
  Retrieval is at recall@10 = 100% and recall@5 = 97% on the benchmark, so contextual retrieval (an LLM call per chunk
  at upload) has little to gain here. RAPTOR and multimodal retrieval have no evidence of need and nothing to evaluate
  them on without a model and diagram-heavy material.
- **The "bounded agentic verify-and-regenerate loop" already exists** as a side effect of the quota design: a rejected
  question leaves a shortfall and the bounded rounds regenerate it. Two tests prove it refills the quiz and that it
  stops (at most `2 x max_rounds` calls) when everything is rejected.
- **Revisit after a real-model run.** If the generation evaluation shows low citation correctness or many unusable
  questions, contextual retrieval and a second regeneration pass are the first things to try.

## Late refactor
- **Split `generate_qa_pairs` again** (`_check_request`, `_choose_context`, `_fill_quota`; complexity 21 to 5). Proved
  identical output to the previous code on 9 seeded scenarios before keeping it.
