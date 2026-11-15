# Evaluation on real documents (2026-11-05)

First test on real PDFs instead of the 8 authored notes in `eval/docs/`. Same code as `main`; only the data differs.

**Documents** (the user's own folder; nothing was modified): *Attention Is All You Need* (11 pages), the Bahdanau attention paper (15), a long LSTM tutorial (42), *Designing Data-Intensive Applications* (491), *Head First Design Patterns* (672). Together: 1,869 chunks at real chunk sizes (the 8 notes gave about 60).

**Model:** `gemini-2.5-flash` on a free-tier key. **Embedder:** the built-in MiniLM.

## 1. Ingestion (no model)

| File | Size | Pages | Chunks | Result |
|---|---|---|---|---|
| Bahdanau | 0.4 MB | 15 | 37 | ok |
| Attention paper | 0.6 MB | 11 | 22 | ok |
| LSTM | 0.7 MB | 42 | 72 | ok |
| DDIA | 4.4 MB | 491 | 891 | **rejected by default**: 491 pages, limit is 300 |
| Head First | 48.5 MB | 672 | 847 | **rejected by default**: 46.2 MB, limit is 25 MB |

The two books load fine once `MAX_PDF_PAGES` and `MAX_UPLOAD_MB` are raised (5 s and 18 s). Extracted text is readable in all five (about 92-94% letters and spaces). The only defect seen is odd spacing in the book ("Y ou've"). The PNG in the folder was not used (OCR is off).

## 2. Retrieval: this changes an earlier conclusion

27 questions I wrote against the five documents, each labelled with a passage that must appear in the retrieved chunk. All 27 gold passages were checked to exist in the extracted text. Six of them occur more than three times in their document, so those are easier.

| Configuration | R@1 | R@5 | MRR [95% CI] | ms/query |
|---|---|---|---|---|
| dense (**current default**) | 0.19 | 0.70 | **0.366** [0.25, 0.50] | 222 |
| hybrid (BM25 + dense) | 0.41 | 0.70 | **0.540** [0.39, 0.70] | 623 |
| dense + reranker | 0.44 | 0.78 | 0.571 [0.42, 0.73] | 1,900 |
| hybrid + reranker | 0.44 | 0.78 | 0.603 [0.47, 0.75] | 2,170 |
| structured chunking, dense / hybrid | 0.19 / 0.33 | 0.52 / 0.67 | 0.318 / 0.448 | 215 / 695 |

Paired changes in MRR (same 27 questions):

| Change | Effect | Verdict |
|---|---|---|
| default -> hybrid | +0.174 [+0.074, +0.288], 15 better / 3 worse | **better** |
| dense -> dense + reranker | +0.205 [+0.061, +0.350], 14 better / 3 worse | **better** |
| hybrid -> hybrid + reranker | +0.063 [-0.084, +0.221], 8 better / 10 worse | no clear difference |
| default -> hybrid + reranker | +0.237 [+0.076, +0.402], 16 better / 6 worse | better |
| structured chunking (any mode) | -0.048 to +0.093, all intervals include 0 | no clear difference |

Hybrid beats dense in 4 of 5 documents (Attention 0.24 -> 0.44, LSTM 0.33 -> 0.58, DDIA 0.37 -> 0.59, Head First 0.41 -> 0.59) and is level on Bahdanau (0.54 vs 0.54).

**What this means.**
- On 8 short notes, hybrid showed "no clear difference" (+0.005). On 1,869 real chunks it is the biggest single gain. The earlier benchmark was too small and too easy to show it. The current default (dense) scores 0.37 here, against 0.89 on the notes, so the notes overstated how well the default works on real documents.
- **Acted on:** retrieval now defaults to `auto`, which uses hybrid for documents of 20+ chunks (see section 2a).
- Hybrid costs about 0.4 s more per query. The reranker costs about 1.7 s more on this CPU (on the notes it was 0.5 s), because it scores more candidates over more text.

### 2a. Does the benefit depend on document size? (one document at a time)

| Document | Chunks | Questions | Dense MRR | Hybrid MRR | Hybrid better / worse |
|---|---|---|---|---|---|
| Attention paper | 22 | 7 | 0.333 | 0.456 | 3 / 0 |
| Bahdanau | 37 | 4 | 0.550 | 0.750 | 1 / 0 |
| LSTM tutorial | 72 | 4 | 0.333 | 0.708 | 4 / 0 |
| Head First | 847 | 6 | 0.414 | 0.586 | 2 / 2 |
| DDIA | 891 | 6 | 0.368 | 0.589 | 4 / 0 |

Hybrid was better or equal at every size tested (22 to 891 chunks) and never lower in score. With only 4-7 questions per row, no single row is conclusive; the point is that there was no size at which dense won. Not tested: documents under 22 chunks (the router sends those to the model whole anyway).

### 2b. Change made: `RETRIEVAL_MODE=auto` is now the default

- `auto` = hybrid when a document has at least `HYBRID_MIN_CHUNKS` (default **20**) chunks, dense below that. `dense` and `hybrid` still work and override the rule.
- 20 sits just under the smallest size tested (22). Documents small enough to fit the model's context are sent whole by the router and never use retrieval, so in practice this means hybrid for long documents.
- Checked on the five PDFs with no mode set: ranks identical to explicit hybrid on every document; average MRR 0.391 (dense) to 0.596 (default).
- Cost: about +0.4 s per topic query on a 1,900-chunk collection (the keyword index is rebuilt per query). Quiz generation without a topic is unaffected.
- Tests: 7 new, 432 total passing; ruff and mypy clean. Four deliberate breaks (always dense, always hybrid, default flipped back, off-by-one on the threshold) were each caught.
- To go back: `RETRIEVAL_MODE=dense`.

**Limits of this result.** 27 questions is small. I wrote them myself, and several contain exact technical terms ("Adadelta", "beam search", "Bloom filter"), which favours keyword matching, so hybrid's advantage is probably larger here than for vague questions from real students. The intervals are wide. Treat the direction as reliable and the size as approximate.

## 3. Question generation with Gemini (partial: quota)

Planned: 20 quizzes (4 question types x 5 documents, 6 questions each). **The free-tier key allows 20 requests per day for this model and ran out partway.** 12 quizzes failed with HTTP 429 `RESOURCE_EXHAUSTED`. Those failures are the API limit, not the app, and are excluded from quality figures below.

Completed: 8 quizzes, 48 questions.

| Measure | Result |
|---|---|
| Model replies that were not parseable JSON | 0 of 20 |
| Items well-formed (the app's own validator) | 70 of 70 |
| Questions returned vs asked, on quizzes that completed | 48 of 48 |
| Answer found in the source text (automatic proxy) | 45 of 48 |
| ...of the 3 misses, actually wrong | 0 (answers were `'1'`, `True` and a paraphrase, which the word-match proxy cannot see) |
| Near-duplicate questions inside one quiz | 2 pairs in 48 questions (about 4%) |
| Page citations spot-checked (Transformer paper, pages 1, 2, 3, 4, 6) | all correct |
| Time per 6-question quiz | about 10-20 s, 2 model calls |
| Tokens, 20 generation calls | about 250,000 in, 10,000 out |
| Estimated cost of those calls | roughly $0.10 at assumed Gemini 2.5 Flash prices (not verified) |

Question types completed: short answer 3 of 5 documents, multiple choice 4 of 5, true/false 1 of 5, **fill in the blank 0 of 5 (never reached)**. So nothing is known yet about the fill-in-the-blank type on a real model.

The quality of the questions was **not** judged by a person beyond the spot checks above. Fill in the manual verdicts in `eval/local/real_gemini.json` (`sample`) to measure citation correctness properly.

## 4. Grading with Gemini (weak evidence)

| Answer given | Graded | Mean score | As expected |
|---|---|---|---|
| The reference answer | 18 of 18 | 1.00 | 18 of 18 |
| "I don't know" | 18 of 18 | 0.00 | 18 of 18 |
| An answer to a different question | **0 of 18 (quota)** | not measured | not measured |

These are the two easiest cases, and the LLM-free pre-checks can settle both without asking the model, so this shows the plumbing works, **not** that grading is accurate. The case that matters (a plausible but wrong or partial answer) was not reached. Grading quality on a real model is still unmeasured.

## 5. Found while doing this

1. **Default limits reject both books** (300 pages, 25 MB). Reasonable for a hosted demo; worth knowing for textbook users.
2. **Real-document retrieval is much weaker than the notes benchmark suggested** (MRR 0.37 vs 0.89).
3. **Near-duplicate questions** still get through inside a quiz (about 4% here); semantic de-duplication is off by default.
4. **The free Gemini tier cannot run this evaluation** in one day: it allows 20 requests per day, and a full run needs roughly 100.

## 6. To finish

- Re-run `eval/local/gen_real.py` after the quota resets (about 12 hours), or with a billed key. Run it in chunks of under 20 requests, or lower `GEMINI_CONTEXT_CHARS` to keep requests small.
- Add 20-30 more real-document questions, written by someone other than the app's author, to firm up the retrieval numbers.
- Decide whether to switch the default to hybrid retrieval for long documents.

## Reproduce

```bash
# queries and results are in eval/local/ (git-ignored: the queries quote short passages from the books)
MAX_UPLOAD_MB=100 MAX_PDF_PAGES=1000 python scripts/eval_retrieval.py \
  --docs-dir <folder of PDFs> --queries eval/local/real_docs_queries.json --scale real
# add: --max-configs 2 --reranker fastembed:Xenova/ms-marco-MiniLM-L-6-v2
```
