# Evaluation

Measures quality, not just code health. Everything here runs without a language model unless noted.

| What | Script | Needs a model? | Data |
|---|---|---|---|
| Retrieval quality (recall@k, MRR; chunking x retrieval mode x embedder x reranker) | `scripts/eval_retrieval.py` | No (embeddings only) | `eval/docs/*.md`, `eval/retrieval_queries.json` (68 queries) |
| Chunk selection (documents touched, chunk similarity) | `scripts/eval_generation.py --provider stub --scope all` | No | same documents |
| Question generation (valid rate, parse failures, grounded rate, coverage, duplicates, calls, tokens, latency) | `scripts/eval_generation.py` | Yes for real numbers (`--provider ollama\|gemini`); `stub` checks the pipeline only | same documents |
| Grading (agreement, error, repeatability, pre-checks) | `scripts/eval_grading.py` | Yes for real numbers; `stub` is a lexical floor | `eval/grading_calibration.json` (33 labelled answers) |
| Compare two runs | `scripts/compare_eval.py before.json after.json` | No | result files |
| Regression gate | `scripts/check_eval.py` | No | `eval/thresholds.json` |

Results are written to `eval/results/` (one JSON per run; generation runs also write a `.review.md` sample for
manual checking of citations and answers).

## Running with a real model
```bash
# Ollama must be running and the model pulled
python scripts/eval_generation.py --provider ollama --model gemma3 --types short,mcq --runs 3
python scripts/eval_grading.py    --provider ollama --model gemma3 --repeats 3 --mode rubric
python scripts/eval_grading.py    --provider ollama --model gemma3 --repeats 3 --mode holistic
# Gemini: set GEMINI_API_KEY
python scripts/eval_generation.py --provider gemini --verify
```
Record the model and date next to the numbers. Fill in the manual verdicts in the `.review.md` file to measure
citation correctness (the script reports only an automatic "grounded" proxy).

## Comparing options
```bash
python scripts/eval_retrieval.py --embedder default --embedder ollama:bge-m3       # embedders
python scripts/eval_retrieval.py --reranker fastembed:Xenova/ms-marco-MiniLM-L-6-v2 --out eval/results/rerank.json
python scripts/compare_eval.py eval/results/<before>.json eval/results/rerank.json
```
`fastembed` is optional: `pip install fastembed`.

## Reading the numbers
- The retrieval corpus is small (8 notes, ~60 chunks, 68 queries) and chunk sizes are scaled down so it behaves like
  long documents. Treat differences smaller than the printed 95% interval as noise; prefer the **paired** comparison.
- A retrieved chunk is relevant when it contains the query's gold snippet. Parent-expansion rows return ~3 chunks per
  hit, so they look better by construction.
- The `stub` generator and grader are deterministic stand-ins. They test the pipeline and give a floor; they say
  nothing about how good a language model is. Their `duplicate_rate` is meaningless (the stub repeats itself).
- The grading labels were written by the benchmark's author. Replace or extend them with your own marking.
