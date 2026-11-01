# Upgrade plan: Exam Q&A Generator

Written 2026-10-31 against `main` @ `37079d0`. Companion to `docs/BASELINE_METRICS.md`: every phase below
names the baseline metric it must improve (or not regress). Effort sizes: **S** = under half a day,
**M** = 1-2 days, **L** = 3+ days.

> Model names, licenses and benchmark numbers below come from secondary web sources (links at the end),
> not primary ones. Verify on the model card and your own data before committing to any of them.

---

## 1. Principles

1. **Measure first.** No retrieval or prompt change ships without a before/after number from the
   evaluation harness (Phase 1). Today we have no quality metrics at all, only code and test metrics.
2. **One change per PR.** Each phase is a separate branch and PR so a regression can be reverted alone.
3. **Coverage beats relevance for this app.** A quiz should span the whole document. Most "RAG
   upgrades" optimise for finding the single best passage for a query, which is the wrong goal when no
   topic is given.
4. **Local-first stays true.** Everything must keep working with Ollama and no paid API. Cloud options
   (Gemini) are additive.
5. **Cheap before clever.** Order phases by impact per effort, not by how interesting they are.

## 2. What the data says about this project

These facts from the baseline shape every recommendation:

- Sample PDFs are **short**: median 4 chunks (about 4,300 characters) per document, 303 chunks over 37 PDFs.
  Most documents fit in one prompt, so retrieval is often unnecessary.
- With no topic filter there is **no retrieval**: all chunks are shuffled and sampled at random.
- With a topic filter it is **dense-only top-30** with no reranker, no keyword search and no similarity cutoff.
- Generation is sequential: 12 LLM calls for a 10-question short-answer quiz (2 generate + 10 grade).
- `llm.py` is 31% covered, `generate_qa_pairs` has complexity 25, and there is no LLM-quality measurement.
- Doc sets are **per quiz and deleted when a new quiz starts**, so there is no long-lived index to migrate
  when the embedder changes.

---

## 3. Recommendations by field

### 3.1 RAG architecture (type)

**Current:** naive RAG (chunk, embed, top-k, stuff into prompt), plus random sampling when no topic.

**Recommended: Modular / adaptive RAG with a router.** One pipeline, three paths chosen per request:

| Path | When | Behaviour |
|---|---|---|
| **A. Full-context** | Document is short enough to fit the model's context (most of your PDFs) | Skip retrieval, pass the whole text. Zero retrieval error, simplest, best coverage |
| **B. Coverage retrieval** | Long document, no topic | Embed chunks, cluster them, take chunks from every cluster so questions span the document |
| **C. Hybrid retrieval + rerank** | Topic filter or "practice weak topics" | BM25 + dense search, fuse, rerank, then generate |

Followed by a **verification step** (a cheap second LLM pass checks each question is answerable from its
cited chunk) before questions reach the student.

**Why this and not the alternatives:**

| Option | Verdict | Reason |
|---|---|---|
| Keep naive RAG | No | Biggest weakness (random sampling) is untouched |
| Long-context only | Partly (Path A) | Fine for short docs; for long docs cost and latency grow and models degrade on very long inputs, so retrieval still matters |
| Agentic RAG (model decides what to retrieve, loops) | Not now | Sources report 2-3x latency; local models are already slow here; a quiz does not need multi-hop reasoning. Revisit only as a bounded verify-and-regenerate loop |
| GraphRAG | No | Heavy graph extraction cost for little gain on single-document study notes |
| RAPTOR (tree of summaries) | Later, optional | Useful for chapter-level questions on very long documents |
| Fine-tuning | No | Not a retrieval problem; no training data |

### 3.2 Retrieval type

**Current:** dense-only vector search (MiniLM), top-k 30 per topic, or random sampling.

**Recommended:**

1. **Hybrid search (BM25 + dense) fused with Reciprocal Rank Fusion** for topic queries. Dense embeddings
   miss exact terms (formulas, acronyms, names); BM25 misses paraphrase. Fusing both is the common 2026
   default. Your doc sets are tiny, so an in-memory BM25 (`rank_bm25`) over the quiz's chunks is enough.
   No search server needed.
2. **A cross-encoder reranker** over the top 20-30 fused results, keeping the best 4-8 for the prompt.
   Rerankers are what turns "plausibly related" into "actually answers the topic".
3. **Coverage selection (clustering or MMR) for no-topic quizzes**, replacing the random shuffle.
4. **A relevance cutoff**: stop returning chunks below a score threshold, so a topic that barely appears
   in the document does not produce questions from irrelevant text.

**Why:** these fix the three measurable defects (random coverage, missed exact terms, no relevance floor)
with small, local, dependency-light changes.

### 3.3 Embedding model

**Current:** Chroma default `all-MiniLM-L6-v2` (small, English, general purpose).

**Recommended:** make the embedder configurable and evaluate, in this order:
1. `qwen3-embedding` 0.6B: the efficiency pick in the sources (strong score for about 1.5 GB), reported Ollama-native.
2. `bge-m3`: strong all-rounder, multilingual, available on Ollama.
3. `nomic-embed-text`: lightest option (about 0.3 GB) if memory is tight.

Keep MiniLM as the zero-setup fallback. Choose by recall@k on your own golden set, not by leaderboard.
Students study in many languages and with equations and code, which is where a stronger, multilingual
embedder helps most. **Migration is trivial** because doc sets are per quiz: give each embedder its own
collection name (`exam_chunks__<embedder>`) so dimensions never clash.

### 3.4 Chunking

**Current:** about 300-word sentence-aware chunks, 50-word overlap, never crossing pages (so median chunk
is about 166 words).

**Recommended:**
- **Structure-aware splitting** on headings and lists, and merge across page boundaries when a paragraph
  continues. Track the heading path as metadata.
- **Token-based sizing** instead of word counts, so chunks match the model's real limits.
- **Parent-child ("small-to-big"):** search small chunks, send the surrounding section to the model.
- **Contextual retrieval (optional, later):** prefix each chunk with a one-sentence LLM summary of where it
  sits in the document before embedding. Sources report large retrieval gains but it costs one LLM call per
  chunk at upload, which is slow on local models.

### 3.5 Document ingestion

**Current:** PDFs with a text layer only; scanned PDFs are rejected.

**Recommended:** OCR fallback for scanned pages; DOCX, PPTX, Markdown and plain text; URL ingestion;
table and heading extraction. Evaluate extraction libraries on your own PDFs (including slides and
two-column papers) before swapping `pypdf`. Multimodal page retrieval for diagram-heavy slides is a later,
optional step.

### 3.6 Question generation

**Current:** batches of 5 over 4 chunks, schema-constrained JSON, one retry, string-match dedupe.

**Recommended:**
- **Answerability check:** second pass confirms the answer is stated in the cited chunk. Drop or regenerate failures.
- **Difficulty and cognitive level** (easy / medium / hard, recall vs. application) as prompt parameters.
- **Answer explanations** with the supporting quote, shown after grading.
- **More question types:** true/false, fill-in-the-blank, "explain why".
- **Semantic dedupe:** compare question embeddings so paraphrases of the same question are caught
  (today only normalised text is compared).
- **Prompt-injection hardening:** delimit PDF text as data and tell the model to ignore instructions in it.
- **Deterministic seeds / lower temperature** for repeatable quizzes.

### 3.7 LLM layer

**Current:** thin `LLMProvider` with Ollama and Gemini; `llm.py` is 31% covered; no timeouts, retries,
temperature, thinking budget or token accounting.

**Recommended:**
- Add `temperature`, `num_ctx` (Ollama's default context window is small, so full-context Path A needs it set explicitly) and request timeouts.
- Pass `response_schema` to Gemini so output is constrained like Ollama's.
- Set a small thinking budget on Gemini for grading to control cost.
- Retry with exponential backoff on transient errors; clear `LLMError` messages.
- Record **tokens and latency per call** so cost and latency metrics become automatic.
- Add an OpenAI-compatible provider (covers LM Studio, vLLM, llama.cpp servers and hosted APIs) behind the same interface.
- Offer a recommended-models table in the README (small and fast for grading, larger for generation).

### 3.8 Grading

**Current:** one LLM call per short answer, free-form 0-1 score, tolerant parser; MCQ deterministic.

**Recommended:**
- `temperature=0` and a fixed prompt version for repeatability.
- **Rubric grading:** have the generator also output 2-4 key points; the grader checks which were hit, and
  the score is computed in code from that, not guessed by the model. This is more consistent and explainable.
- **Cheap pre-checks:** empty or near-identical answers skipped without an LLM call.
- **Parallel grading** and a "retry ungraded answers" button.
- **Calibration set:** about 30 hand-labelled answers to track grading agreement over time.

### 3.9 Performance and cost

**Current:** everything sequential; no caching.

**Recommended:** run independent LLM calls concurrently (generation batches and grading are independent;
cap concurrency for local models); cache LLM responses by prompt hash for repeated runs and tests; reuse
the Chroma client via `st.cache_resource`; stream progress to the UI ("batch 2 of 4"). Concurrency is the
single biggest latency win available with no model change.

### 3.10 Data, persistence and product features

**Current:** results saved to SQLite; quizzes not saved; anonymous per-session id; analytics lost when the
browser session changes.

**Recommended:** persist generated quizzes (retake, share); history page; spaced repetition (resurface
missed questions at 1/3/7 days); optional simple accounts; export to Anki/CSV and a printable exam with
answer key; SQLite schema migrations; TTL cleanup of abandoned doc sets (`delete_doc` currently runs only
when the same session starts a new quiz).

### 3.11 UI and UX

Progress feedback during generation; resume an in-progress quiz after refresh; review mode with the cited
page shown; difficulty and question-type controls; per-topic mastery view; accessibility pass (labels,
keyboard use, contrast). Keep Streamlit unless a phase below proves it limiting.

### 3.12 Security and privacy

Upload limits (file size, page count); file-type validation beyond extension; prompt-injection hardening
(3.6); API keys only from env or session, never logged; non-root Docker user; rate limiting if ever exposed
publicly; a clear note that Gemini mode sends document text to Google while Ollama mode keeps it local.

### 3.13 Testing and quality

`llm.py` contract tests with mocked clients (raise coverage from 31%); golden-file tests for parsers;
property tests for `extract_json_array` and `parse_grade`; the LLM evaluation harness (Phase 1) run on a
schedule, not on every push, because it needs a model; coverage floor in CI; `ruff` and a type checker
(`pyright` or `mypy`); pre-commit hooks.

### 3.14 Code structure

`generate_qa_pairs` (complexity 25) and `qa_generator.py` (maintainability 35) are the hot spots. Split
into small steps: `selection` (which chunks), `prompting`, `parsing/validation`, `dedupe`, `orchestration`.
Replace loose dicts with typed models (`Question`, `GradedAnswer`). Remove the `sys.path` hack in
`ui.py` with a `pyproject.toml` package.

### 3.15 DevOps and docs

Lock file for reproducible installs; slimmer, non-root Docker image; optional `ollama` service in
`docker-compose.yml`; structured logging; LICENSE; README rewrite (Gemini, question types, practice mode,
PDF export, how to run the evaluation); `ARCHITECTURE.md`; `CONTRIBUTING.md`.

---

## 4. Implementation roadmap

Each phase = one branch + one PR. A phase is **done** only when its tests pass, CI is green and its gate
metric is recorded in `docs/BASELINE_METRICS.md` (new column or dated section). Targets marked
*provisional* get fixed after Phase 1 produces real baselines.

### Phase 0: Housekeeping (S)
- **Goal:** clean foundation, nothing functional changes.
- **Tasks:** commit `docs/` and `scripts/`; add `pyproject.toml` (package + tool config) and remove the
  `sys.path` hack; add `ruff` config and fix the 6 current findings; pre-commit; LICENSE; README refresh;
  `gh auth setup-git` so pushes work without workaround.
- **Gate:** lint findings 6 to 0; 94 tests still green; app still starts with `streamlit run`.

### Phase 1: Evaluation harness (L): do this before anything else
- **Goal:** turn every "not measured" row in `BASELINE_METRICS.md` into a number.
- **Tasks:**
  1. `eval/` folder with 8-10 representative PDFs (public-domain or your own) covering short notes, a long
     document, a scanned-text-free slide deck and a table-heavy doc.
  2. **Retrieval set:** about 40 (query, correct page) pairs, hand-written, to compute recall@k and MRR.
  3. **Generation metrics:** valid-question rate, retry rate, MCQ validity, duplicate rate, topic/page coverage.
  4. **Citation check:** sample 30 questions and verify the cited page contains the answer (manual first, LLM-judge later).
  5. **Grading set:** about 30 labelled answers (correct / partial / wrong) for agreement and repeatability.
  6. **Instrumentation:** token counts and latency per call written to a JSON report.
  7. `scripts/run_eval.py --model <name>` writes `eval/results/<date>-<model>.json`; a script diffs two runs.
- **Needs:** Ollama server running with a pulled model (and optionally a Gemini key). Decide the reference model.
- **Gate:** all section-6 baseline rows filled in, with model and date recorded.

### Phase 2: LLM-layer quick wins (S-M)
- **Tasks:** `temperature`, `num_ctx`, timeouts; Gemini `response_schema`; Gemini thinking budget for
  grading; retry with backoff; token and latency logging; prompt-injection delimiters in generation and
  grading prompts; mocked-client tests for both providers.
- **Gate:** `llm.py` coverage 31% to 85%+ *(provisional)*; grading repeatability variance drops; no drop in
  valid-question rate.

### Phase 3: Refactor the generation pipeline (M)
- **Why now:** Phases 4-8 all change this code; splitting it first keeps each later PR small.
- **Tasks:** extract `selection.py` (choose chunks), `prompting.py`, `validation.py`, `dedupe.py`; add typed
  `Question` model; keep `generate_qa_pairs` as a thin orchestrator; no behaviour change.
- **Gate:** identical test results; complexity of `generate_qa_pairs` 25 to under 10; maintainability of `qa_generator` improves.

### Phase 4: Coverage-aware chunk selection (M): biggest quality win
- **Tasks:** embed chunks (already done by Chroma), cluster with k-means (k about n/4, capped), draw batches
  round-robin from clusters; MMR as a lighter alternative; keep random as a config fallback; track which
  chunks and pages each quiz used.
- **Gate:** topic/page coverage per quiz up by a clear margin *(provisional: +30% distinct pages)*; duplicate rate not worse; latency unchanged.

### Phase 5: Embedder upgrade (M)
- **Tasks:** `EMBEDDING_MODEL` setting; embedder factory (MiniLM default, Ollama models via embedding
  function); per-embedder collection name; evaluate `qwen3-embedding` 0.6B, `bge-m3`, `nomic-embed-text`
  on the retrieval set; pick a default.
- **Gate:** recall@k and MRR improve over MiniLM *(provisional: recall@5 +10 points)*; ingest time and memory recorded and acceptable.

### Phase 6: Chunking upgrade (M)
- **Tasks:** heading-aware splitter with heading-path metadata; merge across pages; token-based sizing;
  parent-child retrieval; re-run retrieval and generation eval with each option behind a flag.
- **Gate:** retrieval metrics up or flat, citation correctness up, chunk-size distribution closer to target.

### Phase 7: Hybrid retrieval + reranker for topic quizzes (M-L)
- **Tasks:** in-memory BM25 per doc set; Reciprocal Rank Fusion with dense results; relevance cutoff;
  cross-encoder reranker behind an interface (`bge-reranker-v2-m3` as the permissive-license baseline;
  `qwen3-reranker` only after a license check); config flags to turn each stage off; latency budget check.
- **Gate:** topic-query recall@k and MRR up *(provisional: MRR +0.1)*; added latency under 1 s per query for the typical doc; questions for an absent topic now fail gracefully ("topic not found in your notes").

### Phase 8: Adaptive router and full-context path (M)
- **Tasks:** estimate document tokens; if it fits the chosen model's context with headroom, use Path A;
  else route to Path B or C by whether a topic was given; surface the chosen path in logs and (optionally) UI.
- **Gate:** for short documents, valid-question rate and citation correctness not worse, fewer LLM calls; long documents unchanged.

### Phase 9: Question quality: verifier, difficulty, explanations (M-L)
- **Tasks:** answerability verifier (second LLM pass, batch several questions per call to limit cost);
  semantic dedupe using embeddings; difficulty parameter; explanation field with supporting quote;
  new question types behind flags.
- **Gate:** citation correctness up *(provisional: 90%+ on the 30-sample check)*; duplicate rate down; added cost and latency recorded and accepted.

### Phase 10: Concurrency, caching and progress (M)
- **Tasks:** bounded thread pool for generation batches and grading; response cache keyed by prompt hash;
  `st.cache_resource` for the Chroma client; per-batch progress in the UI.
- **Gate:** end-to-end quiz latency down substantially *(provisional: 2-3x on Gemini; smaller on a single local model)*; no change in quality metrics; no race conditions in tests.

### Phase 11: Grading upgrade (M)
- **Tasks:** generator emits key points; rubric-based scoring in code; pre-checks; parallel grading;
  "retry ungraded" button; calibration set wired into the harness.
- **Gate:** agreement with labelled set up and repeatability variance near zero; grading cost per quiz down or flat.

### Phase 12: Ingestion formats and OCR (M-L)
- **Tasks:** OCR fallback for scanned PDFs; DOCX/PPTX/MD/TXT loaders; URL loader; upload limits and
  validation; extraction-library bake-off on your PDFs.
- **Gate:** scanned PDFs no longer rejected; extraction quality compared on the eval set; limits enforced.

### Phase 13: Persistence and product features (L)
- **Tasks:** save quizzes; history and retake; spaced-repetition scheduler; Anki/CSV and printable exports;
  schema migrations; TTL cleanup job; optional accounts.
- **Gate:** features covered by tests; data survives restart and container rebuild.

### Phase 14: DevOps and hardening (M)
- **Tasks:** lock file; slim non-root image; optional `ollama` service in compose; structured logging;
  coverage floor and type checking in CI; scheduled eval workflow; ARCHITECTURE and CONTRIBUTING docs.
- **Gate:** clean-machine build and first run documented and working; image size recorded vs. baseline.

### Phase 15: Optional advanced work (decide after Phase 9)
Contextual retrieval (summary-prefixed chunks); RAPTOR-style chapter summaries; multimodal page retrieval
for diagram-heavy slides; bounded agentic verify-and-regenerate loop. Each only if the evaluation shows a
remaining quality gap worth its cost.

---

## 5. Order and dependencies

```
0 Housekeeping
└─ 1 Evaluation harness  ← everything below is gated on this
   ├─ 2 LLM quick wins
   └─ 3 Pipeline refactor
      ├─ 4 Coverage selection ─┐
      ├─ 5 Embedder ───────────┤
      ├─ 6 Chunking ───────────┼─ 7 Hybrid + rerank ─ 8 Adaptive router ─ 9 Verifier/difficulty
      │                        │
      ├─ 10 Concurrency/cache (independent, any time after 3)
      └─ 11 Grading (independent, after 2)
12 Ingestion · 13 Product features · 14 DevOps  (parallel tracks, any time after 0)
15 Advanced (only if eval shows a gap)
```

**Suggested first month:** Phases 0, 1, 2, 3, then 4. That fixes the largest weakness (random coverage) with
measurements to prove it.

## 6. Risks and open decisions

| Item | Risk / question | Mitigation |
|---|---|---|
| Reference model | Results depend heavily on which LLM runs the eval | Decide one Ollama model (and optionally Gemini) as the yardstick; record it with every run |
| Eval set size | Too small a set makes gains look noisy | Keep sets small but fixed, report counts, avoid over-tuning to them |
| Reranker license | Sources disagree on Qwen3-Reranker's license | Default to the permissively licensed BGE reranker; check the model card before using Qwen |
| Memory | Larger embedders and rerankers compete with the LLM for RAM/VRAM | Measure memory in Phase 5/7; keep the small model as default if needed |
| Latency | Verifier, reranker and rubric grading add calls | Record added latency per phase; combine with Phase 10 concurrency |
| Cloud privacy | Gemini mode uploads document text | State it in the UI and README |
| Scope creep | Phase 13 features are open-ended | Ship each as its own small PR; defer accounts |

## 7. Tracking checklist

- [x] 0 Housekeeping (LICENSE and `gh auth setup-git` left for the owner: both are your decision/config)
- [ ] 1 Evaluation harness (fills all "not measured" baseline rows)
- [x] 2 LLM quick wins (live Ollama/Gemini runs and grading-repeatability measurement still pending: need a model)
- [x] 3 Pipeline refactor
- [x] 4 Coverage-aware selection (measured with a stub LLM; question-quality effect needs a live model)
- [x] 5 Embedder upgrade (mechanism done; choosing the best model by recall@k is pending Phase 1)
- [x] 6 Chunking upgrade (behind `CHUNKING=structured`; default stays `legacy` until the eval shows it helps)
- [x] 7 Hybrid retrieval + reranker (behind flags; defaults chosen with the Phase 1 retrieval eval)
- [x] 8 Adaptive router
- [x] 9 Verifier, difficulty, explanations (offline-tested; effect on real question quality needs a live model)
- [x] 10 Concurrency, caching, progress
- [x] 11 Grading upgrade (agreement/repeatability need a live model and the calibration set from Phase 1)
- [x] 12 Ingestion formats and OCR (real OCR path untested here: no Tesseract installed)
- [ ] 13 Persistence and product features
- [ ] 14 DevOps and hardening
- [ ] 15 Optional advanced work

## 8. Sources

- Embeddings: [Best Local Embedding Models for RAG (2026)](https://d-central.tech/local-embedding-models/), [10 Best Embedding Models in 2026](https://dev.to/misterankit/10-best-embedding-models-powering-ai-systems-in-2026-ch8), [13 Best Embedding Models in 2026](https://elephas.app/blog/best-embedding-models.md)
- Rerankers: [Best open weight reranker models 2026](https://presenc.ai/research/best-open-weight-reranker-models-2026), [Best rerankers for RAG 2026](https://futureagi.com/blog/best-rerankers-for-rag-2026/), [Top 5 Reranking Models](https://machinelearningmastery.com/?p=23793)
- RAG patterns: [RAG in 2026: naive to agentic](https://dev.to/agdex_ai/rag-in-2026-from-naive-retrieval-to-agentic-rag-a-complete-implementation-guide-48lh), [RAG is not dead: advanced retrieval patterns](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo), [RAG architecture overview](https://futureagi.com/blog/rag-architecture-llm-2025/)
