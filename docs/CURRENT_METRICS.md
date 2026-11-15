# Current metrics, compared with the original baseline

| | Baseline (`docs/BASELINE_METRICS.md`, sections 1-5) | Current (this document) |
|---|---|---|
| Code | `main` @ `37079d0` | `main` @ `f9ff649` (after PRs #2 and #3) |
| Measured | 2026-10-31 | 2026-11-02 |
| Environment | Python 3.13.2, macOS (Darwin 24.6, Intel) | same |

**Every "Current" number below was measured again on the final code**, not copied from the progress log. Where the
progress log and a fresh measurement differ, this document is the one to trust (see section 9).
"Verdict" is **better**, **worse**, **same** or **new** (no baseline existed).

---

## 1. Code size

| Metric | Baseline | Current | Verdict |
|---|---|---|---|
| App source (`app/`), lines / source lines | 821 / 621 | 3,206 / 2,342 | 3.8x larger |
| Tests, lines / source lines | 604 / 452 | 3,113 / 2,395 | 5.2x larger |
| Test-to-code ratio (source lines) | 0.73 | 1.02 | better |
| Modules in `app/` (incl. `__init__`) | 11 | 35 | more, smaller modules |
| Largest app files (lines) | `qa_generator` 252, `ui` 173 | `ui` 295, `loaders` 261, `llm` 205 | `qa_generator` is no longer among the largest |
| Test files | 7 | 21 | |
| Direct dependencies | 8 | 10 (`numpy`, `fonttools` added) | +2 |
| Tracked files | 29 | 109 | |
| `.git` size (local clone) | 5.0 MB | 6.9 MB | +1.9 MB (this clone still holds the 15 old local phase branches, so it overstates the repository) |
| Comments per source line | 4% | 5% | slightly better |

## 2. Tests and CI

| Metric | Baseline | Current | Verdict |
|---|---|---|---|
| Tests | 94 passed | **425 passed**, 0 failed, 0 skipped | better |
| Runtime, plain / with coverage | ~12 s / ~32 s | **58 s / 112 s** | **worse (about 5x slower)** |
| Statement coverage | 89% (552 statements, 56 missed) | **94.7%** (2,102 statements, 101 missed) | better |
| Branch coverage | 178 branches, 18 partial | 642 branches, 33 partial | similar share |
| Files at 100% coverage | not recorded | 19 of 35 | new |
| `llm.py` coverage | 31% | 99% | better |
| `ui.py` coverage | 81% | 83% | slightly better |
| Files below 95% | `ui` 81, `llm` 31, `vector_store` 89, `report` 93 | `models` 0 (type definitions only), `ocr` 50, `ui` 83, `rerank` 86, `loaders` 89, `embeddings` 93 | different gaps |
| CI jobs | tests on Python 3.11 + 3.13 | lint, types (mypy), tests on 3.11 + 3.13 with a 92% coverage floor, Docker build/health (main only), weekly evaluation | better |
| CI result | passing | passing on PR #3 (Docker job skipped on pull requests by design) | same |

Why the suite is slower: many new tests ingest real documents with the real embedding model, run Streamlit
`AppTest` flows, start subprocesses (the Streamlit Cloud launch test) or compare timings of parallel calls. The
slowest three are about 9 s, 7 s and 5 s each.

## 3. Code quality

| Metric | Baseline | Current | Verdict |
|---|---|---|---|
| Average cyclomatic complexity | 4.28 (A) over 46 blocks | 4.19 (A) over 207 blocks | slightly better |
| Blocks at grade C or worse | 4 | **12** | **worse** |
| Grade D blocks | 1 (`generate_qa_pairs`, 25) | **3**: `heading_level` 23, `validate_pairs` 23, `retrieve` 21 | **worse** |
| `generate_qa_pairs` complexity | 25 (D) | 5 (A) | better |
| Lowest maintainability index | `qa_generator` 35.2, `ui` 44.2 | `ui` 31.9, `loaders` 33.7, `llm` 41.4 (`qa_generator` now ~60) | `ui` worse |
| Lint, baseline rules (`ruff --isolated --select E,F`) | 6 findings | 20 findings, **all `E402` in `ui.py`** (the deliberate path bootstrap from PR #3) | see note |
| Lint, project config | not configured | 0 findings | new |
| Type errors (mypy) | not checked | 0 across 35 files, enforced in CI | new |

Lint note: the 20 findings are one pattern, a `sys.path` line that must run before the imports so the app starts
under `streamlit run app/ui.py` on hosts that cannot install the package. The project config allows it for that one
file; the baseline's stricter isolated run does not know about the exception.

## 4. Chunking and retrieval

Defaults did not change, so the unchanged rows are expected.

| Metric (37 sample PDFs, default settings) | Baseline | Current | Verdict |
|---|---|---|---|
| PDFs / chunks / empty | 37 / 303 / 0 | 37 / 303 / 0 | same |
| Chunk size in characters (mean / median / p95 / max) | 1065 / 1126 / 1663 / 2211 | 1065 / 1126 / 1663 / 2211 | same |
| Chunk size in words (mean / median) | 166 / 161 | 166 / 161 | same |
| Chunks that fit the prompt per 4-chunk batch | 3.99 | 3.99 | same |
| Opt-in structured chunking (5 distinct PDFs, tokens) | did not exist | 163 chunks, range 82-399, median 218 (legacy: 169 chunks, range 3-553, median 282) | new |

| Capability | Baseline | Current |
|---|---|---|
| Retrieval type (default) | naive dense search; random sampling with no topic | same default, plus opt-in hybrid (BM25 + dense), optional reranker, relevance cutoff |
| Embedding model | built-in MiniLM only | MiniLM (default), `ollama:<model>` or `fastembed:<model>` |
| No-topic chunk choice | random shuffle | source-aware coverage selection (round-robin over files, clusters within each file); `random` still available |
| Short documents | 4 of every 5 chunks could be left out of a call | whole document sent when it fits the budget (router) |
| Topic absent from the notes | not detected | clear error (with a distance cutoff set) |
| Parent expansion (small-to-big) | none | opt-in |

## 5. Quality benchmark (new: nothing was measured before)

Corpus: 8 authored notes, 68 retrieval queries, 33 labelled grading answers. Chunk sizes scaled down so the short notes
behave like long documents. Treat differences inside the printed interval as noise.

### 5.1 Retrieval (MiniLM, no language model needed)

| Configuration | R@1 | R@5 | MRR | Paired change in MRR vs default |
|---|---|---|---|---|
| legacy chunks + dense **(current default)** | 0.81 | 0.97 | **0.888** [0.83, 0.94] | baseline |
| legacy + hybrid | 0.81 | 1.00 | 0.893 | +0.005, no clear difference |
| structured + dense | 0.75 | 0.96 | 0.849 | -0.039, no clear difference |
| structured + hybrid | 0.82 | 0.97 | 0.888 | -0.001, no clear difference |
| default + **reranker** (`Xenova/ms-marco-MiniLM-L-6-v2`) | 0.90 | 1.00 | **0.942** [0.90, 0.98] | +0.054 [-0.007, +0.114], consistent (12 better / 4 worse queries) |

Parent-expansion rows are left out: they return about three chunks per hit, so they look better by construction.
The reranker costs about +0.5 s per topic query (277 ms to 804 ms here) and needs the optional `fastembed` package, so it
stays opt-in. The default embedder takes 277 ms per query; `fastembed:BAAI/bge-small-en-v1.5` takes about 13 ms with a
similar score (0.871).

### 5.2 Chunk selection (no-model stand-in, 120 runs per row, 8 files)

| Strategy | Files touched per quiz | Chunk similarity (lower = more varied) |
|---|---|---|
| random (the baseline behaviour) | 72.5% | 0.126 |
| source-aware coverage (current default) | **94.5%** | **0.103** |

Difference in files touched: +22 points, 95% interval [+19.8, +24.3].

### 5.3 Grading (no-model lexical stand-in, 33 labelled answers)

| Mode | Agreement with labels | Mean absolute error | Repeatability (std of repeated scores) |
|---|---|---|---|
| rubric (default) | 64% | 0.207 | 0.000 |
| holistic | 61% | 0.208 | 0.000 |

The stand-in is a floor, not a model. LLM-free pre-checks settle 3 of 33 answers, all three correctly.

### 5.4 Generation (no-model stand-in, pipeline only)

Valid rate 1.00, parse failures 0.00, grounded 1.00 (trivially true for an extractive stub), 2.0 LLM calls per
8-question quiz. These show the pipeline works end to end; they say nothing about a language model's quality.

## 6. Cost and latency

| Metric | Baseline | Current | Verdict |
|---|---|---|---|
| LLM calls, 10-question short-answer quiz | 12 (2 generate + 10 grade) | 12 | same |
| Calls for multiple-choice / true-false grading | 0 | 0 | same |
| Generation prompt size (short / MCQ / mixed, characters) | 4,952 / 5,168 / 5,060 | 5,386 / 5,450 / 5,418 (+8.8% / +5.5% / +7.1%) | slightly worse |
| Grading prompt size (characters) | 276 | 487 (+76%) | worse |
| Estimated input tokens per 10-question quiz | about 3,170 | about 3,910 (+23%) | worse |
| Estimated cost on Gemini 2.5 Flash | about $0.005-0.01 | about the same (the extra input costs roughly $0.0002); output size is the unknown | about the same |
| Concurrency | none (sequential) | bounded parallel: Gemini 4, Ollama 1 by default | better |
| Simulated quiz time (0.5 s per call, 12 calls) | 6.0 s | 6.0 s at 1 worker; 3.0 s at 2; 2.0 s at 4; 1.5 s at 8 | better when allowed |
| Real generation / grading latency | not measured | **still not measured** (no model run) | open |

Prompts grew because they now carry the prompt-injection guard, the explanation and key-point instructions, and the
rubric. Output tokens probably grew too (key points and explanations); that is not measured. The cost figures use the
same method and price assumptions as the baseline (third-party price listings, not verified against the provider).

## 7. Deployment and operations

| Metric | Baseline | Current | Verdict |
|---|---|---|---|
| Docker image size | 1.92 GB | **1.60 GB** (embedding model included) | better (-17%) |
| Runs as | root | uid 1000 (non-root) | better |
| Build tools / curl in the image | yes | no | better |
| Dependency versions | loose bounds | pinned `requirements.lock` (Python 3.11) | better |
| Health check | needs curl | Python one-liner | better |
| Time to healthy | not measured | 7 s | new |
| Works with no internet | no (downloads an 80 MB model on first upload, 120 s here) | yes (first ingestion 2.4 s offline) | better |
| Data location | `./data` bind mount (root-owned on Linux) | named volume | better |
| Optional bundled Ollama | no | `--profile ollama` | new |
| Logging | text | text or JSON (`LOG_FORMAT=json`) | better |
| Launch with plain `streamlit run app/ui.py` (Streamlit Cloud) | worked | worked, **broke in PR #2**, fixed in PR #3, now covered by a test | same, after an incident |

## 8. Capabilities

| Area | Baseline | Current |
|---|---|---|
| Question types | short answer, multiple choice | + true/false, fill in the blank |
| Quiz options | count, topic | + difficulty, explanations, optional answerability check |
| Inputs | PDF with a text layer | + Word, PowerPoint, Markdown, text, web pages (private addresses refused), optional OCR; size and page limits |
| Grading | LLM score for short answers | + key-point rubric (score computed in code), LLM-free pre-checks, retry for ungraded answers |
| Persistence | results table | + versioned database, saved quizzes and retake, spaced-repetition queue, optional profile name |
| Exports | results PDF | + printable exam (with answer key), Anki file, CSV |
| Data control | none | delete-my-data button, documents expire after 7 days |
| Evaluation | none | retrieval, selection, generation and grading benchmarks, comparison tool, regression gate, weekly workflow |
| Documentation | README | + architecture, contributing, upgrade plan, metrics, decision log, evaluation guide |

## 9. What got worse, and corrections

Worse than the baseline:
1. **Test suite is about 5x slower** (58 s vs 12 s). Fine for CI, noticeable for local iteration.
2. **More complex code in places:** 12 functions at grade C or worse (was 4) and three at grade D (`heading_level`,
   `validate_pairs`, `retrieve`). The one function that was grade D before (`generate_qa_pairs`) is now grade A.
3. **`ui.py` is the least maintainable file** (index 31.9, was 44.2): it grew from 173 to 295 lines.
4. **Prompts are larger**, especially grading prompts (+76%), so input cost per quiz rises about 23%.
5. **Larger surface to maintain:** 3.8x the code, 2 more dependencies, 109 tracked files (was 29).
6. **A launch-path regression reached the deployed app** (PR #2 removed the path bootstrap; Streamlit Cloud then
   crashed). Fixed in PR #3 with a regression test; the earlier checks had not exercised that launch path.

Corrections to numbers in the progress log (`BASELINE_METRICS.md` sections 7-10):
- Its final snapshot said app code was 3,198 / 2,337 lines and tests 424. Fresh measurement after PR #3: 3,206 / 2,342
  and 425 (the extra test and bootstrap lines come from PR #3).
- Its "worst functions" row listed `heading_level` and `retrieve` as the grade D functions. A fresh complexity run also
  shows `validate_pairs` at grade D (23), which that table omitted.
- Its lint row said 0 findings; that is true for the project config, but under the baseline's strict isolated rules the
  current count is 20 (all the one intentional `E402` pattern).

## 10. Still not measured (needs a language model or the owner)

Valid-question rate, parse-failure and retry rate, MCQ validity, citation correctness, duplicate rate, real grading
agreement and repeatability, verifier effect, real latency and token cost; Ollama embedders; the larger `bge-reranker`
models; real OCR with Tesseract; the Docker CI job and weekly evaluation workflow on GitHub (they have not run there).

## 11. How these were measured

| Metric group | Command |
|---|---|
| Size, complexity, maintainability | `uvx radon raw -s app tests`, `radon cc -s -a app`, `radon mi -s app` |
| Lint / types | `ruff check --isolated --select E,F --line-length 120 app tests`, `ruff check app tests scripts`, `mypy app` |
| Tests and coverage | `pytest -q`, `pytest -q --cov` |
| Chunking | `python scripts/chunk_stats.py` |
| Retrieval, reranker, selection, grading, generation | `scripts/eval_retrieval.py`, `eval_grading.py`, `eval_generation.py` (see `eval/README.md`) |
| Simulated latency | `scripts/latency_sim.py` |
| Prompt sizes | a stub model on one sample PDF, as in the baseline |
| Docker | `docker build`, `docker run`, `docker images` |
