"""Generate quiz question/answer pairs from ingested PDF chunks."""
from __future__ import annotations

import io
import json
import logging
import random
import re
from typing import Dict, List, Optional

from .llm import LLMProvider
from .pdf_loader import load_pdf
from .vector_store import add_chunks, file_hash, get_all_chunks, similarity_search

log = logging.getLogger(__name__)

BATCH_SIZE = 5            # questions requested per LLM call
CHUNKS_PER_BATCH = 4      # distinct context chunks per call
MAX_CONTEXT_CHARS = 6000

_SYSTEM_PROMPT = """You are a professional exam tutor. Based ONLY on the context below, \
generate exam questions.

Format (output nothing else, no markdown):
{"questions": [{"question": "What is ...?", "answer": "...", "topic": "ShortTopicName"}]}
"""

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"},
                       "topic": {"type": "string"}},
        "required": ["question", "answer", "topic"]}}},
    "required": ["questions"],
}


def ingest_pdf(source, doc_id: str, filename: str = "upload.pdf", data: Optional[bytes] = None) -> int:
    """Chunk a PDF and store it under doc_id. `source` is a path or file-like.
    Re-ingesting identical bytes is idempotent."""
    if data is None:
        data = source.read() if hasattr(source, "read") else open(source, "rb").read()
    chunks = load_pdf(io.BytesIO(data))
    return add_chunks(chunks, doc_id, source=filename, file_id=file_hash(data))


def extract_json_array(text: str) -> list:
    """Pull a JSON array out of an LLM reply (tolerates prose, code fences, a
    top-level object wrapping the list, curly quotes and trailing commas)."""
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"```(?:json)?", "", text)
    candidates = []
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        cand = re.sub(r",\s*([\]}])", r"\1", cand)  # trailing commas
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            if "question" in parsed:  # model returned a single bare question object
                return [parsed]
    raise ValueError("No valid JSON array found in model response.")


def validate_pairs(items: list) -> List[Dict[str, str]]:
    """Keep only well-formed items; default a missing topic to 'General'."""
    good = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = str(it.get("question", "")).strip()
        a = str(it.get("answer", "")).strip()
        if not q or not a:
            continue
        good.append({"question": q, "answer": a,
                     "topic": str(it.get("topic") or "General").strip() or "General"})
    return good


def _dedupe(pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen, out = set(), []
    for p in pairs:
        key = re.sub(r"\W+", " ", p["question"].lower()).strip()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _generate_batch(llm: LLMProvider, chunks: List[str], n: int, topic: Optional[str]) -> List[Dict[str, str]]:
    context = "\n\n".join(chunks)[:MAX_CONTEXT_CHARS]
    instruction = (f"Generate {n} question-answer pairs focused on the topic '{topic}'."
                   if topic else f"Generate {n} varied exam-style question-answer pairs.")
    prompt = f"{_SYSTEM_PROMPT}\n{instruction}\n\nContext:\n{context}"
    for attempt in (1, 2):  # one retry on unparseable output
        raw = llm.generate(prompt, schema=QUESTIONS_SCHEMA)
        log.debug("LLM raw response: %s", raw)
        try:
            return validate_pairs(extract_json_array(raw))
        except ValueError:
            log.warning("Unparseable model output (attempt %d)", attempt)
    return []


def generate_qa_pairs(doc_id: str, llm: LLMProvider, n: int = 10,
                      topic: Optional[str] = None) -> List[Dict[str, str]]:
    """Generate up to n validated, de-duplicated Q&A pairs, in batches drawn
    from different chunks so coverage is broad."""
    chunks = similarity_search(topic, k=50, doc_id=doc_id) if topic else get_all_chunks(doc_id)
    if not chunks:
        raise ValueError(f"No content found for document set: {doc_id}")

    random.shuffle(chunks)
    pairs: List[Dict[str, str]] = []
    pos, rounds = 0, 0
    while len(pairs) < n and rounds < 2 * (n // BATCH_SIZE + 1):
        batch = [chunks[(pos + i) % len(chunks)] for i in range(min(CHUNKS_PER_BATCH, len(chunks)))]
        pos += CHUNKS_PER_BATCH
        rounds += 1
        pairs = _dedupe(pairs + _generate_batch(llm, batch, min(BATCH_SIZE, n - len(pairs) + 1), topic))
    if not pairs:
        raise ValueError("The model did not return any usable questions. Try again or use a different model.")
    return pairs[:n]
