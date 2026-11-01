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

## 6. Not measured yet (needs a running LLM)
Ollama is installed but its server was not running, so these are open. Measure them before changing
retrieval or prompts, and record the model and date:

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
