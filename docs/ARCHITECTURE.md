# Architecture

A Streamlit app that turns study material into quizzes with an LLM, grades the answers, and tracks progress.
Everything runs locally by default (Ollama + an on-device embedding model); Gemini is an optional hosted LLM.

## Flow

```
upload / URL ──► loaders ──► chunking ──► embeddings ──► vector store (Chroma, one doc set per quiz)
                                                              │
   topic? ──► retrieval (dense | hybrid + rerank) ────────────┤
                                                              ▼
                        router ─► full | topic | coverage ─► selection ─► prompting ─► LLM
                                                                                        │
              validation ◄─ parse JSON ◄────────────────────────────────────────────────┘
                  │  (optional verifier, dedupe)
                  ▼
              questions ──► UI quiz ──► grader (MCQ/TF exact; short/cloze: rubric or holistic via LLM)
                                              │
                          results DB · saved quizzes · spaced repetition · exports
```

## Modules (`app/`)

| Area | Module | Responsibility |
|---|---|---|
| UI | `ui.py` | Streamlit page: inputs, quiz form, results, history sidebar |
| Ingestion | `loaders.py` | PDF/DOCX/PPTX/Markdown/text/URL to chunks; size/page limits; URL safety |
| | `pdf_loader.py`, `chunking.py` | Legacy and structure-aware chunkers; `Chunk` type |
| | `ocr.py` | Optional OCR for scanned PDFs |
| Retrieval | `embeddings.py` | Embedder choice (`default` MiniLM or `ollama:<model>`), one collection per embedder |
| | `vector_store.py` | Chroma wrapper (add, search, load all, delete) |
| | `bm25.py`, `retrieval.py`, `rerank.py` | Keyword ranking, rank fusion, optional reranker |
| | `clustering.py`, `selection.py` | k-means and the chunk-selection strategies (random, coverage, full) |
| | `router.py` | Chooses full / topic / coverage per quiz from the document size and provider budget |
| Generation | `qa_generator.py` | Orchestrates a quiz; `ingest_*` entry points |
| | `planning.py`, `prompting.py`, `validation.py`, `dedupe.py`, `verifier.py`, `prompt_safety.py` | Quotas, prompts and schemas, output checks, de-duplication, optional answerability check, prompt-injection delimiters |
| LLM | `llm.py` | Ollama and Gemini providers: retries, usage stats, options |
| | `llm_cache.py`, `concurrency.py` | Optional response cache; bounded parallel calls |
| Grading | `grader.py`, `recommendation.py` | Scoring, rubric, pre-checks, retry; weak-topic picker |
| Persistence | `database.py` | SQLite with numbered migrations; results |
| | `quizzes.py`, `srs.py`, `maintenance.py` | Saved quizzes, spaced repetition, document expiry |
| Output | `report.py`, `exports.py` | Results PDF; exam PDF, Anki, CSV |
| Support | `config.py`, `models.py`, `logging_config.py` | Environment config, types, logging |

## Design rules

- **Defaults are conservative.** New behaviour ships behind a setting; the old behaviour stays the default until the
  evaluation (`docs/UPGRADE_PLAN.md`, Phase 1) shows a change helps. See `docs/DECISIONS.md` for why each choice.
- **Untrusted text is data.** PDF text, web pages and student answers are wrapped in delimiters and neutralised
  before they enter a prompt (`prompt_safety.py`). Web fetching refuses private addresses.
- **LLM failures never crash a quiz.** Providers raise `LLMError`; the UI shows a message. Ungradable answers are
  kept as "not graded" (score `None`) instead of a misleading zero and can be retried.
- **Per-quiz document sets.** Each quiz ingests into its own `doc_id`; sets expire after `DOC_TTL_DAYS`.
- **Everything optional is lazy.** OCR, reranking and the Ollama embedder import their packages only when enabled.

## Configuration

All settings are environment variables read once in `config.py`; `.env.example` lists them with comments.
