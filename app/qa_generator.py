"""Generate quiz questions (short-answer and multiple-choice) from ingested PDFs.

This module orchestrates the pipeline; the pieces live in `selection` (which chunks),
`prompting`, `validation` (parse + check LLM output), `dedupe` and `planning` (quotas/rounds).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

from .dedupe import dedupe, semantic_dedupe
from .embeddings import get_embedding_function
from .llm import LLMProvider
from .loaders import load_document, load_web_page
from .concurrency import effective_workers, run_parallel
from .planning import BATCH_SIZE, QUESTION_TYPES, max_rounds, plan_wave, trim_to_quota, type_quota
from .prompting import DIFFICULTIES, MAX_CONTEXT_CHARS, build_prompt, schema_for
from . import config
from .router import FULL, choose_path, context_budget
from .selection import CHUNKS_PER_BATCH, FullContextCycler, gather, make_cycler, normalize_topics
from .validation import N_OPTIONS, extract_json_array, match_option, validate_pairs
from .verifier import verify_pairs
from .vector_store import add_chunks, file_hash, get_all_chunks

log = logging.getLogger(__name__)

_match_option = match_option  # kept for existing imports

__all__ = [
    "BATCH_SIZE", "CHUNKS_PER_BATCH", "MAX_CONTEXT_CHARS", "N_OPTIONS", "QUESTION_TYPES",
    "extract_json_array", "generate_qa_pairs", "ingest_document", "ingest_pdf", "ingest_url", "validate_pairs",
]


def ingest_document(source, doc_id: str, filename: str = "upload.pdf", data: Optional[bytes] = None) -> int:
    """Chunk an uploaded document (PDF, DOCX, PPTX, Markdown or text) and store it under doc_id.
    `source` is a path or file-like, or None when `data` is given. Re-ingesting identical bytes is
    idempotent. Raises UploadError for unsupported/too large files."""
    if data is None:
        data = source.read() if hasattr(source, "read") else open(source, "rb").read()
    chunks = load_document(data, filename)
    return add_chunks(chunks, doc_id, source=filename, file_id=file_hash(data))


ingest_pdf = ingest_document  # original name, kept for existing callers


def ingest_url(url: str, doc_id: str) -> int:
    """Download a web page (or PDF) and store its text under doc_id."""
    chunks, name = load_web_page(url)
    return add_chunks(chunks, doc_id, source=name, file_id=file_hash(url.encode("utf-8")))


def _generate_batch(llm: LLMProvider, hits: List[dict], n: int, topic: Optional[str], qtype: str,
                    max_chars: Optional[int] = None, difficulty: Optional[str] = None,
                    verify: bool = False) -> List[Dict]:
    prompt = build_prompt(hits, n, topic, qtype, max_chars, difficulty)
    for attempt in (1, 2):  # one retry on unparseable output
        raw = llm.generate(prompt, schema=schema_for(qtype))
        log.debug("LLM raw response: %s", raw)
        try:
            pairs = validate_pairs(extract_json_array(raw), hits, qtype, keep_chunk=True)
            if verify and pairs:
                pairs = verify_pairs(llm, pairs, hits)
            return pairs
        except ValueError:
            log.warning("Unparseable model output (attempt %d)", attempt)
    return []


def generate_qa_pairs(doc_id: str, llm: LLMProvider, n: int = 10,
                      topic: Union[None, str, Sequence[str]] = None,
                      question_type: str = "short", strategy: Optional[str] = None,
                      on_route: Optional[Callable[[str], None]] = None,
                      difficulty: Optional[str] = None, verify: Optional[bool] = None,
                      on_progress: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
    """Generate up to n validated, de-duplicated questions in batches drawn from
    different chunks. `topic` may be one string or a list of topics.
    question_type: "short", "mcq", or "mixed" (alternates batches).
    strategy: how chunks are chosen, "coverage" or "random" (default from SELECTION_STRATEGY).
    on_route: optional callback told which path was used ("full", "topic" or "coverage").
    difficulty: None, "easy", "medium" or "hard". verify: second-pass check that each question is supported
    by its passage (default from VERIFY_QUESTIONS). Types "tf" (True/False) and "cloze" (fill in the blank).
    on_progress(done, wanted): called after each round with the number of usable questions so far.
    Independent calls run in parallel up to the provider's concurrency limit (1 = sequential)."""
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    verify = config.VERIFY_QUESTIONS if verify is None else verify
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {QUESTION_TYPES}")
    topics = normalize_topics(topic)
    topic_text = ", ".join(topics) or None

    all_hits = get_all_chunks(doc_id)
    if not all_hits:
        raise ValueError(f"No content found for document set: {doc_id}")
    budget = context_budget(llm)
    path = choose_path(all_hits, topics, budget)
    # A topic must still exist in the notes; retrieval doubles as that relevance check.
    hits = gather(doc_id, topics) if topics else all_hits
    if topics and not hits:
        raise ValueError(f"The topic '{topic_text}' was not found in your notes. "
                         "Try a different topic, or leave it empty to cover everything.")
    log.info("Quiz routing: %s path (%d chunks, budget %d chars)", path, len(all_hits), budget)
    if on_route:
        on_route(path)

    quota = type_quota(question_type, n)
    if path == FULL:
        chunks, per_call, max_chars = FullContextCycler(all_hits), config.FULL_CONTEXT_BATCH, budget
    else:
        chunks, per_call, max_chars = make_cycler(hits, strategy), BATCH_SIZE, None
    pairs: List[Dict] = []
    failed: Set[str] = set()  # types whose batches returned nothing; don't let them starve the others
    workers = effective_workers(llm)
    rounds, limit = 0, max_rounds(n)
    while rounds < limit:
        wave = plan_wave(quota, pairs, failed, per_call, min(workers, limit - rounds))
        if not wave:
            break
        jobs = [(qtype, count, chunks.next_batch()) for qtype, count in wave]  # selection stays single-threaded
        results = run_parallel(
            lambda job: _generate_batch(llm, job[2], job[1], topic_text, job[0], max_chars, difficulty, verify),
            jobs, workers)
        for (qtype, _, _), new in zip(jobs, results):
            rounds += 1
            if not new:
                failed.add(qtype)
            pairs = dedupe(pairs + new)
            if config.SEMANTIC_DEDUPE:
                pairs = semantic_dedupe(pairs, get_embedding_function(), config.SEMANTIC_DEDUPE_THRESHOLD)
        if on_progress:
            on_progress(min(len(pairs), n), n)
    if not pairs:
        raise ValueError("The model did not return any usable questions. Try again or use a different model.")
    final = trim_to_quota(pairs, quota, question_type)
    for p in final:
        p.pop("_chunk", None)  # internal: index of the cited passage, only needed by the verifier
    return final
