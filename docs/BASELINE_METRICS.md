# Baseline metrics

Measured on `main` @ `37079d0` (2026-10-31), Python 3.13.2, macOS (Darwin 24.6).
Reproduce the static/test numbers with `scripts/metrics.sh`. Compare after any upgrade or feature.

## 1. Code size
| Metric | Value |
|---|---|
| App source (`app/`) LOC / SLOC | 821 / 621 |
| Tests (`tests/`) LOC / SLOC | 604 / 452 |
| Test-to-code ratio (SLOC) | 0.73 |
| Python modules in `app/` | 11 (`qa_generator.py` 252 lines, `ui.py` 173 lines are the largest) |
| Direct dependencies | 8 (`requirements.txt`) |
| Tracked files | 29 |
| `.git` size | 5.0 MB |

## 2. Tests
| Metric | Value |
|---|---|
| Tests | 94 passed, 0 failed, 0 skipped |
| Test files | 7 |
| Runtime (plain / with coverage) | ~12 s / ~32 s |
| CI | GitHub Actions, Python 3.11 + 3.13, both passing |
| Statement coverage | **89%** (552 stmts, 56 missed) |
| Branch coverage | 178 branches, 18 partial |

Per-module coverage:

| Module | Cover | Notes |
|---|---|---|
| `config`, `database`, `recommendation`, `__init__` | 100% | |
| `pdf_loader` | 98% | |
| `grader`, `qa_generator` | 96% | |
| `report` | 93% | |
| `vector_store` | 89% | |
| `ui` | 81% | Streamlit paths: Gemini key branch, analytics, error handlers |
| `llm` | **31%** | Real Ollama/Gemini calls are untested (tests use a fake LLM) |

## 3. Code quality
| Metric | Value |
|---|---|
| Avg cyclomatic complexity | 4.28 (grade A) over 46 blocks |
| Blocks at complexity C or worse | 4: `generate_qa_pairs` 25 (D), `parse_grade` 12, `extract_json_array` 12, `validate_pairs` 12 |
| Maintainability index | all files A; lowest: `qa_generator` 35.2, `ui` 44.2; highest: `recommendation` 70.7 |
| Lint (ruff `--isolated --select E,F`, line length 120) | 6 findings (4x E702, 1x E731, 1x F401) |
| Comment ratio | ~3% (docstrings excluded) |

## 4. Chunking and retrieval (37 sample PDFs in `tmp/`)
| Metric | Value |
|---|---|
| PDFs / empty (no text layer) | 37 / 0 |
| Total chunks | 303 (median 4 per PDF, so these are short documents) |
| Chunk size, characters | mean 1065, median 1126, p95 1663, max 2211 |
| Chunk size, words | mean 166, median 161 (target is 300) |
| Chunks that fit in the prompt per 4-chunk batch | mean 3.99; 70 of 71 batches fit all 4 |
| Embedding model | Chroma default `all-MiniLM-L6-v2` (ONNX) |
| Retrieval | dense only, no rerank, no hybrid. Without a topic: random sampling |

Note: median chunk size is about half the 300-word target because chunks never cross pages.

### Retrieval type
| Aspect | Current |
|---|---|
| Overall type | Naive single-stage dense (semantic) vector retrieval |
| Embedding | Bi-encoder `all-MiniLM-L6-v2` (ONNX via Chroma default), English, general-purpose |
| Index / search | Chroma persistent collection (approximate nearest neighbour), Chroma default distance metric (not set explicitly) |
| Scope filter | Metadata filter `doc_id` (one document set per quiz) |
| With a topic filter | One query per topic, top-k = 30 each; results merged, de-duplicated by text, shuffled |
| Without a topic filter | No query: all chunks of the quiz are loaded and shuffled (random sampling) |
| Context per LLM call | 4 chunks per batch, cycled through the shuffled list; 6,000-character cap |
| Hybrid / keyword (BM25) | No |
| Reranker | No |
| Query rewriting / expansion | No |
| Diversity selection (MMR, clustering) | No |
| Parent-document / small-to-big | No |
| Similarity threshold | No (always returns k results, relevant or not) |
| Citations | LLM returns the chunk number; mapped back to source file and page |
| Retrieval quality metrics (recall@k, MRR) | **not measured** (no labelled query set yet) |

## 5. Cost and latency

### LLM calls and prompt size (measured with a stub LLM on a sample PDF, best case: no retries)
| Step | Calls | Input per call | Output per call (stub) |
|---|---|---|---|
| Generate 10 questions (short / MCQ / mixed) | 2 | ~4,950 / ~5,170 / ~5,060 chars (~1,250 tokens) | ~485 / ~690 / ~590 chars (~120-170 tokens) |
| Grade one short answer | 1 per answer | ~276 chars (~70 tokens) | ~72 chars (~20 tokens) |
| Grade one multiple-choice answer | 0 (deterministic) | none | none |
| 10-question short-answer quiz, end to end | 12 (2 generate + 10 grade) | | |

Tokens are estimated at ~4 characters per token. Real models write longer answers than the stub, so
real output is probably 300-600 tokens per generation call. All calls run sequentially.

### Cost per 10-question short-answer quiz (estimate)
| Provider | Price | Estimated cost |
|---|---|---|
| Ollama (local) | $0 API fees; cost is your own hardware | $0 |
| Gemini 2.5 Flash (`GEMINI_MODEL` default) | $0.30 / 1M input, $2.50 / 1M output (thinking tokens bill as output) | ~$0.005-0.01, before thinking tokens |
| Gemini 2.5 Flash-Lite | $0.10 / 1M input, $0.40 / 1M output | not estimated |

Prices come from third-party aggregators citing Google's page, retrieved 2026-10-31. Verify at
ai.google.dev/gemini-api/docs/pricing. No thinking budget is set in `GeminiProvider`, so output
tokens (and cost) may be several times higher than the estimate. Re-measure with real token counts.

### Latency
| Metric | Value |
|---|---|
| Slowest test (ingest PDF + full pipeline, stub LLM) | 4.6 s, includes embedding-model load |
| Slowest UI test (page load) | 2.2 s |
| Ingest + embed a PDF | not measured separately |
| Time to generate 10 questions (Ollama / Gemini) | **not measured** |
| Time to grade one short answer (Ollama / Gemini) | **not measured** |
| End-to-end quiz (generate + grade 10) | **not measured** |
| Concurrency | none; generation batches and grading calls are sequential |

Latency was not measured because no LLM was available. Expected shape: total time is roughly the
number of calls (12 for a 10-question short-answer quiz) times the time per call.

## 6. Still open: needs a language model
Ollama was not used in this build, so the rows below that need a real model are open. The harness for each is
ready (`eval/README.md`); run it and record the model and date. Rows marked *stub* were measured with the
no-model stand-ins, which test the pipeline, not a model:

| Metric | How to measure |
|---|---|
| Valid-question rate | Valid questions / requested, over N runs per PDF |
| Parse-failure and retry rate | Count of batches needing the retry in `_generate_batch` |
| MCQ validity | Share of MCQs with 4 distinct options including the answer |
| Citation correctness | Share of questions whose cited page actually contains the answer (spot-check 30) |
| Topic coverage | Distinct source pages used / total pages, per quiz |
| Duplicate rate | Near-duplicate questions across 3 quizzes from the same PDF |
| Grading consistency | Variance of score when grading the same answer 5 times |
| Grading accuracy | Agreement with a hand-labelled set of ~30 answers |
| Latency and real token usage | Fill in the "not measured" rows and cost estimate in section 5 |
| Docker image size and cold start | `docker images`, time to healthy |

## 7. Phase log (metrics after each upgrade phase)

| Phase | Date | Result vs. baseline |
|---|---|---|
| 0 Housekeeping | 2026-10-31 | Lint (E,F) 6 to 0. Tests 94 passed (unchanged). `sys.path` hack removed; `pyproject.toml` added (wheel builds); app starts and `/_stcore/health` returns ok |
| 2 LLM-layer quick wins | 2026-10-31 | Tests 94 to 114. `llm.py` coverage 31% to 99%; total coverage 89% to 93%. Added: temperature, `num_ctx`, timeouts, retry with backoff on transient errors, Gemini `response_json_schema`, optional Gemini thinking budget (`GEMINI_THINKING_BUDGET`, off by default), per-call token/latency stats, prompt-injection delimiters, grading at temperature 0. **Not yet measured with a live model:** repeatability variance, valid-question rate |
| 3 Pipeline refactor | 2026-10-31 | No behaviour change: old vs. new output identical on 9 seeded scenarios (3 question types x 3 topic modes, incl. retry path). `generate_qa_pairs` complexity 25 (D) to 8 (B); `qa_generator.py` maintainability 35.2 to 69.5; average complexity 4.28 to 3.72. New modules: `selection`, `planning`, `prompting`, `validation`, `dedupe`, `models` |
| 4 Coverage-aware selection | 2026-11-01 | Stub-LLM measurement, 30 seeds, 169 chunks from 5 distinct documents (`scripts/coverage_measure.py`). **n=10:** distinct documents per quiz 1.4 to 1.9 (+36%); mean pairwise similarity of chunks used 0.226 to 0.169 (-25%, more diverse). **n=25:** 1.8 to 2.5 (+39%); 0.232 to 0.190 (-18%). Unique chunks used and LLM calls unchanged; clustering overhead 6 ms at 169 chunks, 287 ms at 5,000. Tests 127 to 141. Caveats: few distinct documents/pages in the sample; effect on real question quality not measured (needs a live model). Default strategy `coverage`; `SELECTION_STRATEGY=random` reverts |
| 5 Embedder upgrade | 2026-11-01 | `EMBEDDING_MODEL` = `default` or `ollama:<model>`; separate Chroma collection per embedder (default keeps the name `exam_chunks`, so existing stores keep working); store cache now keyed by (path, embedder). Tests 141 to 155. **Not measured:** recall@k / MRR per model and ingest time/memory (need Ollama + the Phase 1 retrieval set), so the default stays MiniLM |
| 6 Chunking upgrade | 2026-11-01 | New `CHUNKING=structured` (heading-aware, token-sized, page-spanning) and `PARENT_EXPANSION` (small-to-big). 5 distinct sample PDFs, estimated tokens/chunk. **Legacy:** 169 chunks, mean 259 / median 282, range 3-553, 0 headings, 0 page-spanning. **Structured:** 163 chunks, mean 233 / median 218, range 82-399, 161 with a heading, 80 page-spanning. Result: much tighter size distribution and no tiny fragments, but the median is **not** closer to the 400-token target (sections in these PDFs are short). Tests 155 to 181. **Not measured:** retrieval metrics and citation correctness (need Phase 1). Default stays `legacy` |
| 7 Hybrid retrieval + reranker | 2026-11-01 | `RETRIEVAL_MODE=dense|hybrid` (in-memory BM25 + dense, Reciprocal Rank Fusion), optional `RERANKER=fastembed:<model>` (optional dependency, untested against a real model here), `RETRIEVAL_MAX_DISTANCE` cutoff, clear error for a topic absent from the notes. Tests 181 to 204. Default remains `dense` (no behaviour change). **Not measured yet:** recall@k / MRR and added latency; the Phase 1 retrieval eval (needs no LLM) will decide the defaults |
| 8 Adaptive router | 2026-11-01 | `ADAPTIVE_ROUTING=on` (default): if the whole document fits the budget (`FULL_CONTEXT_MAX_CHARS` 6000; Gemini 120,000 via `GEMINI_CONTEXT_CHARS`; Ollama `2 x OLLAMA_NUM_CTX` when set) every call sees all of it. On the 5 distinct sample documents (stub LLM): 4 take the **full** path, the 156-chunk one stays on **coverage**. A 5-chunk document that previously had 1 of 5 chunks left out of every call now sends all 5. Calls for a 10-question quiz unchanged (2) with the stub because it always returns 5 questions per call; `FULL_CONTEXT_BATCH` (default 5) can raise per-call questions, but the effect on call count and question quality is **not measured** (needs a live model). Topic quizzes on short documents still run retrieval first so an absent topic is reported. Tests 204 to 218. UI shows which path was used |
| 9 Question quality | 2026-11-01 | Added: answerability verifier (`VERIFY_QUESTIONS`, off by default, fails open, 1 extra call per batch), difficulty (`easy/medium/hard`), per-question `explanation` (shown after grading and in the PDF), true/false (`tf`, graded without an LLM) and fill-in-the-blank (`cloze`, LLM-graded) question types, optional embedding-based paraphrase removal (`SEMANTIC_DEDUPE`, off). UI gained difficulty and the two new types. Tests 218 to 244. Threshold check on the real embedder (8 paraphrase + 8 related pairs, small sample): 0.9 catches 3/8 paraphrases and wrongly drops 1/8; **0.85 catches 7/8, wrongly drops 1/8**, so the default is 0.85. **Not measured (needs a live model):** citation correctness with/without the verifier (plan target 90%+), verifier cost/latency, explanation quality |
| 10 Concurrency, caching, progress | 2026-11-01 | Parallel generation waves (`plan_wave`) and parallel grading (`grade_all`), bounded per provider: Gemini 4, Ollama 1 (default, unchanged), hard cap 16. Optional on-disk response cache (`LLM_CACHE_DIR`). Per-batch generation progress and parallel grading progress in the UI. **Sequential output unchanged:** identical to the previous commit on 9 seeded scenarios. **Simulated latency** (fake model sleeping 0.5 s/call, 12 calls for a 10-question short-answer quiz, `scripts/latency_sim.py`): workers 1 / 2 / 4 / 8 = 6.0 s / 3.0 s / 2.0 s / 1.5 s (x1.0 / x2.0 / x3.0 / x4.0). Real gains depend on the provider's rate limits and, for Ollama, on `OLLAMA_NUM_PARALLEL` and hardware: **not measured on a real model**. Tests 244 to 270. Chroma client reuse needed no change (the store cache is already process-wide) |
| 11 Grading upgrade | 2026-11-01 | Rubric grading (`GRADING_MODE=rubric`, default): the generator emits 2-4 `key_points` per short question; the grader returns one true/false per point and the score is `covered / total`, computed in code (partial credit, shown as a checklist in the UI). LLM-free pre-checks: blank, "I don't know"-style and exact/~95%-similar answers. "Retry ungraded answers" button (only re-grades failures and stores the new scores). Parallel grading from Phase 10. Falls back to holistic when a question has no key points. Tests 270 to 302. **Not measured (needs a live model + labelled set):** agreement with human labels, score variance on repeated grading, share of answers skipped by pre-checks, grading cost per quiz |
| 12 Ingestion formats and OCR | 2026-11-01 | New inputs: DOCX and PPTX (standard library only, zip-bomb guarded), Markdown/text (BOM-aware decoding), web pages and PDF links (http/https only; localhost, private, link-local and metadata addresses refused; redirects refused; size and time limits), upload size (`MAX_UPLOAD_MB` 25) and PDF page (`MAX_PDF_PAGES` 300) limits, content-based format detection. Optional OCR fallback for text-less PDFs (`pypdfium2` + `pytesseract` + Tesseract; `OCR_MODE=off|auto`). Documents without real pages use page 0, shown without a page number. UI accepts all formats plus a URL box. Tests 302 to 350. **Extraction bake-off** (5 distinct sample PDFs, `scripts/extraction_bakeoff.py`): pypdf 2.55 s / 166,556 chars / 11 glued words; pypdfium2 0.21 s / 166,948 chars / 11 glued; pdfplumber 7.55 s / 160,071 chars / 326 glued. PyMuPDF could not be built in this environment (SSL error), so it was not compared. `fonttools` added to requirements (pypdf warns without it; sample output identical either way). **Not tested:** the real OCR path (no Tesseract here; only an injected fake) and fetching real websites |
| 13 Persistence and product features | 2026-11-01 | SQLite schema now versioned with numbered migrations (`PRAGMA user_version`, v2; verified on a pre-versioning database with data kept). New: saved quizzes + retake + delete, history with attempts/average score, spaced repetition (simplified SM-2, 1/3/N days, ease 1.3-2.8), exports (printable exam PDF, exam + answer key, Anki TSV, CSV), TTL cleanup of old documents (`DOC_TTL_DAYS`=7, run once per browser session), per-profile "delete my data", optional profile name (history survives refresh). Tests 350 to 378. Bugs found and fixed by tests: list-typed topic broke saving practice quizzes; sidebar history was drawn before the new quiz existed. **Not done:** real accounts/passwords (a profile name is not security), cross-device sync |
| 14 DevOps and hardening | 2026-11-01 | **Image (built and run with Docker here):** old Dockerfile 1.92 GB; new 1.60 GB (-17%) *including* the embedding model baked in. Runs as uid 1000 (non-root); no compiler or curl; health check via Python; pinned `requirements.lock` (371 lines, Python 3.11 universal). Verified: `/_stcore/health` ok, **healthy 7 s after start**, ingestion + quiz generation work with the network disabled (first ingestion 1.4 s; before baking the model in it took 120 s and needed internet). Compose: named data volume (no bind-mount permission problems), optional `ollama` service (`--profile ollama`). Structured logging (`LOG_FORMAT=json`). **Type checking:** mypy found 23 errors in 7 files; all fixed, now 0 across 35 files and enforced in CI. CI: separate lint, types, tests (3.11 + 3.13, coverage floor 92%) and a docker build/health job on pushes to main. **Coverage 94.6%** (was 89%), 382 tests. Added `docs/ARCHITECTURE.md` and `CONTRIBUTING.md`. **Not done:** LICENSE (owner's choice), running the CI workflows on GitHub (nothing was pushed) |
| 15 Optional advanced work | 2026-11-01 | **Decided not to build any of it now**, on evidence from Phase 1. Contextual retrieval: retrieval is already at recall@10 100% / recall@5 97% on the benchmark, so there is little to gain for one extra LLM call per chunk. RAPTOR, multimodal: no evidence of need and nothing to evaluate them on without a model and diagram-heavy material. Bounded verify-and-regenerate loop: **already exists** (rejected questions leave a quota shortfall that the bounded generation rounds refill; proven by a test). Revisit after a real-model run |
| 1 Evaluation harness | 2026-11-01 | See section 8. Tests 424, `generate_qa_pairs` complexity brought back from 21 to 5 (outputs identical to before on 9 seeded scenarios) |

## 8. Evaluation results (Phase 1, 2026-11-01)

Corpus: 8 authored study notes (~2,800 words), 68 retrieval queries (38 paraphrase, 10 exact-term, 20 hard/indirect),
33 hand-labelled grading answers. Chunk sizes scaled down so the short notes behave like long documents.
Reproduce with `eval/README.md`; raw results are in `eval/results/`.

### 8.1 Retrieval (MiniLM default embedder, no model needed)
| Configuration | R@1 | R@5 | MRR | 95% CI | Paired change in MRR vs current default |
|---|---|---|---|---|---|
| legacy chunks + dense **(current default)** | 0.81 | 0.97 | 0.888 | [0.83, 0.94] | baseline |
| legacy + hybrid (BM25 + dense) | 0.81 | 1.00 | 0.893 | [0.84, 0.94] | +0.005 [-0.054, +0.062] no clear difference |
| structured chunks + dense | 0.75 | 0.96 | 0.849 | [0.79, 0.91] | -0.039 [-0.089, +0.008] no clear difference |
| structured + hybrid | 0.82 | 0.97 | 0.888 | [0.83, 0.95] | -0.001 [-0.078, +0.074] no clear difference |
| structured + dense + parent* | 0.87 | 0.99 | 0.919 | [0.87, 0.97] | +0.031 [-0.034, +0.096] no clear difference |
| structured + hybrid + parent* | 0.85 | 1.00 | 0.910 | [0.86, 0.96] | +0.021 [-0.057, +0.096] no clear difference |

\* parent rows return ~3 chunks of text per hit, so containing the gold snippet is easier by construction.
**Conclusion: no chunking or retrieval-mode change is distinguishable from the current default on this benchmark,
so all defaults stay.** Hybrid helps exact-term queries slightly (MRR 0.933 to 0.950) and hurts hard ones slightly
(0.779 to 0.752); neither is significant.

### 8.2 Reranker (real cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed)
| Configuration | MRR without | with | Paired change [95% CI] | better / worse queries | ms per query |
|---|---|---|---|---|---|
| legacy + dense (default) | 0.888 | 0.942 | +0.054 [-0.007, +0.114] | 12 / 4 | 239 to 992 |
| legacy + hybrid | 0.893 | 0.942 | +0.049 [+0.007, +0.101] **better** | 10 / 2 | 298 to 749 |
| structured + dense | 0.849 | 0.936 | +0.087 [+0.028, +0.147] **better** | 15 / 4 | 297 to 831 |
| structured + hybrid | 0.888 | 0.936 | +0.049 [+0.004, +0.099] **better** | 10 / 2 | 309 to 894 |

The reranker is the only change with statistical support (4 of 6 configurations clearly better, all positive;
hard-query MRR 0.78 to 0.86). Cost: roughly +0.5 to 0.75 s per topic query and an optional dependency
(`pip install fastembed`, ~80 MB model). It stays opt-in (`RERANKER=fastembed:...`). This was also the first run of
the reranker code against a real model (previously only a fake).

### 8.3 Embedder
| Embedder | MRR (legacy + dense) | ms per query |
|---|---|---|
| default MiniLM (Chroma ONNX) | 0.888 | 239 |
| `fastembed:BAAI/bge-small-en-v1.5` | 0.871 | 13 |

No clear quality difference; bge-small is ~18x faster per query. Ollama embedders (`bge-m3`, `nomic-embed-text`,
`qwen3-embedding`) were not run: **open**.

### 8.4 Chunk selection (stub; 120 runs per arm, 8 documents, 57 chunks)
| Strategy | Documents touched per quiz | Mean similarity of chunks shown (lower = more varied) |
|---|---|---|
| random (original) | 0.725 | 0.126 |
| clusters only (Phase 4 as first shipped) | 0.695 | 0.097 |
| **source-aware clusters (now the default)** | **0.945** | 0.103 |

Source-aware vs random: +0.220 documents [+0.198, +0.243]; vs clusters-only: +0.250 [+0.228, +0.272]. See section 9.

### 8.5 Grading (stub lexical grader = floor; 33 labelled answers)
| Mode | Agreement with labels | Mean abs error | Mean score correct / partial / wrong | Repeatability (std) |
|---|---|---|---|---|
| rubric (default) | 64% (21/33) | 0.207 | 0.80 / 0.13 / 0.08 | 0.000 |
| holistic | 61% (20/33) | 0.208 | 0.76 / 0.18 / 0.09 | 0.000 |

LLM-free pre-checks settle 3 of 33 answers (blank, "I don't know", exact copy), all 3 correctly. The lexical floor fails
mostly on *partial* answers; a real model must beat it there. Real-model agreement and repeatability: **open**.

### 8.6 Generation (stub; pipeline only)
valid_rate 1.00, parse_fail_rate 0.00, item_validity 1.00, grounded_rate 1.00 (all trivially true for an extractive
stub), 2.0 LLM calls per 8-question quiz, section_coverage 0.66. `duplicate_rate` is meaningless for the stub (it repeats
itself). The same script run with `--provider ollama|gemini` produces the real numbers: **open**.

### 8.7 Still open (needs a real model or the owner)
Valid-question rate, parse-failure and retry rate, MCQ validity, citation correctness (manual review file), duplicate
rate, grading agreement/accuracy/repeatability, verifier effect, real latency and token cost; Ollama embedders; the
`bge-reranker` family (large downloads not run); real OCR with Tesseract.

## 9. Corrections to earlier claims in this log
- **Phase 4** reported "+36-39% distinct documents" for clustering alone. On the balanced 8-document benchmark with
  120 runs per arm, clustering alone *reduced* documents touched (-0.030, CI [-0.054, -0.007]) while making chunks
  less similar (-0.029). The earlier sample had one 156-chunk document and four small ones, where random sampling is
  dominated by the big one. The source-aware version added in Phase 1 fixes the weakness (+0.25 documents).
- **Phase 5** shipped Ollama/fastembed embedders that crashed on every search: Chroma calls `embed_query`, which my
  embedders lacked (tests only covered adding chunks). Found by running a real model; fixed with regression tests.
- **Phase 6/7** combination "hybrid + parent expansion" silently dropped the dense half of retrieval (MRR 0.80). Found
  by the benchmark; fixed (expansion now happens after ranking) with a regression test.
- **Phase 3** gate (`generate_qa_pairs` complexity < 10) had been undone by later phases (21); restored to 5.
- Two measurement bugs in my own evaluation tools were found and fixed during Phase 1 (a stub grader that rated
  everything correct because it parsed the prompt's instructions as the student answer, and a document matcher that
  ignored Markdown heading markers). Both are covered by tests.

## 10. Final snapshot versus the original baseline
| Metric | Baseline (section 1-5) | Now |
|---|---|---|
| App code, lines / source lines | 821 / 621 | 3,198 / 2,337 |
| Test code, lines / source lines | 604 / 452 | 3,098 / 2,384 |
| Modules in `app/` (incl. `__init__`) | 11 | 35 |
| Tests | 94 | **424** |
| Statement coverage | 89% | **94.7%** (branch partials 32 of 640) |
| `llm.py` coverage | 31% | 99% |
| Average cyclomatic complexity | 4.28 | 4.19 (204 blocks, was 46) |
| Worst functions | `generate_qa_pairs` 25 (D) | `heading_level` 23 (D), `retrieve` 21 (D); `generate_qa_pairs` 5 (A) |
| Lowest maintainability index | `qa_generator` 35.2 | `ui.py` 32.0, `loaders.py` 33.7 |
| Lint findings (E,F) | 6 | 0 |
| Type errors (mypy) | not checked | 0 across 35 files (enforced in CI) |
| Direct dependencies | 8 | 10 (added `numpy`, `fonttools`) |
| Docker image | 1.92 GB, root, builds from loose versions | 1.60 GB, non-root, pinned, runs offline |
| Question types | short, multiple choice | + true/false, fill-in-the-blank |
| Inputs | PDF with a text layer | + DOCX, PPTX, Markdown, text, web pages, optional OCR |
| LLM calls (10-question short quiz) | 12 sequential | 12, parallel where the provider allows (Gemini x4) |
| Retrieval MRR / recall@5 (benchmark) | not measured | 0.888 / 0.97 (default); 0.942 / 1.00 with reranker |
| Documents touched by a multi-file quiz (benchmark) | not measured | 0.725 random, now 0.945 |
