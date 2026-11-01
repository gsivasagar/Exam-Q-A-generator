"""Generate quiz questions (short-answer and multiple-choice) from ingested PDFs.

This module orchestrates the pipeline; the pieces live in `selection` (which chunks),
`prompting`, `validation` (parse + check LLM output), `dedupe` and `planning` (quotas/rounds).
"""
from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Sequence, Union

from .dedupe import dedupe
from .llm import LLMProvider
from .pdf_loader import load_pdf
from .planning import BATCH_SIZE, QUESTION_TYPES, max_rounds, next_request, trim_to_quota, type_quota
from .prompting import MAX_CONTEXT_CHARS, build_prompt, schema_for
from .selection import CHUNKS_PER_BATCH, gather, make_cycler, normalize_topics
from .validation import N_OPTIONS, extract_json_array, match_option, validate_pairs
from .vector_store import add_chunks, file_hash

log = logging.getLogger(__name__)

_match_option = match_option  # kept for existing imports

__all__ = [
    "BATCH_SIZE", "CHUNKS_PER_BATCH", "MAX_CONTEXT_CHARS", "N_OPTIONS", "QUESTION_TYPES",
    "extract_json_array", "generate_qa_pairs", "ingest_pdf", "validate_pairs",
]


def ingest_pdf(source, doc_id: str, filename: str = "upload.pdf", data: Optional[bytes] = None) -> int:
    """Chunk a PDF and store it under doc_id. `source` is a path or file-like.
    Re-ingesting identical bytes is idempotent."""
    if data is None:
        data = source.read() if hasattr(source, "read") else open(source, "rb").read()
    chunks = load_pdf(io.BytesIO(data))
    return add_chunks(chunks, doc_id, source=filename, file_id=file_hash(data))


def _generate_batch(llm: LLMProvider, hits: List[dict], n: int, topic: Optional[str], qtype: str) -> List[Dict]:
    prompt = build_prompt(hits, n, topic, qtype)
    for attempt in (1, 2):  # one retry on unparseable output
        raw = llm.generate(prompt, schema=schema_for(qtype))
        log.debug("LLM raw response: %s", raw)
        try:
            return validate_pairs(extract_json_array(raw), hits, qtype)
        except ValueError:
            log.warning("Unparseable model output (attempt %d)", attempt)
    return []


def generate_qa_pairs(doc_id: str, llm: LLMProvider, n: int = 10,
                      topic: Union[None, str, Sequence[str]] = None,
                      question_type: str = "short", strategy: Optional[str] = None) -> List[Dict]:
    """Generate up to n validated, de-duplicated questions in batches drawn from
    different chunks. `topic` may be one string or a list of topics.
    question_type: "short", "mcq", or "mixed" (alternates batches).
    strategy: how chunks are chosen, "coverage" or "random" (default from SELECTION_STRATEGY)."""
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {QUESTION_TYPES}")
    topics = normalize_topics(topic)
    topic_text = ", ".join(topics) or None

    hits = gather(doc_id, topics)
    if not hits:
        raise ValueError(f"No content found for document set: {doc_id}")

    quota = type_quota(question_type, n)
    chunks = make_cycler(hits, strategy)
    pairs: List[Dict] = []
    failed = set()  # types whose batches returned nothing; don't let them starve the others
    for _ in range(max_rounds(n)):
        request = next_request(quota, pairs, failed)
        if request is None:
            break
        qtype, shortfall = request
        new = _generate_batch(llm, chunks.next_batch(), min(BATCH_SIZE, shortfall + 1), topic_text, qtype)
        if not new:
            failed.add(qtype)
        pairs = dedupe(pairs + new)
    if not pairs:
        raise ValueError("The model did not return any usable questions. Try again or use a different model.")
    return trim_to_quota(pairs, quota, question_type)
