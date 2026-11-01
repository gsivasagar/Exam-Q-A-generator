"""Structure-aware chunking: respects headings, may span pages, sized in tokens.

Used when CHUNKING=structured (the legacy word-count chunker in pdf_loader stays the default until the
evaluation harness shows the structured one is better).
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

_STOP = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "vs", "with"}
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+(\S.{0,90})$")
_KEYWORD = re.compile(r"^(chapter|section|unit|part|lecture|module|topic)\s+(\d+|[ivxlc]+)\b[\s.:\-]*(.{0,80})$", re.I)
_ENDS_SENTENCE = re.compile(r"[.!?,;:]$")


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token); good enough for sizing chunks."""
    return math.ceil(len(text) / 4) if text else 0


def heading_level(line: str) -> Optional[int]:
    """Return a heading depth (1 = top) if `line` looks like a heading, else None.

    Heuristics only (PDF text has no real structure): numbered ("2.1 Cell Biology"), keyword
    ("Chapter 3: Energy"), ALL CAPS short lines, and short Title Case lines without end punctuation."""
    s = line.strip()
    if not s or len(s) > 100 or _ENDS_SENTENCE.search(s):
        return None
    m = _NUMBERED.match(s)
    if m and not re.fullmatch(r"[\d.\s]+", m.group(2)):
        rest = m.group(2).split()
        if len(rest) <= 12 and rest[0][0].isupper():
            return m.group(1).count(".") + 1
    if _KEYWORD.match(s):
        return 1
    letters = [c for c in s if c.isalpha()]
    words = s.split()
    if len(letters) >= 3 and s.upper() == s and len(words) <= 10:
        return 1
    if 1 < len(words) <= 6:
        content = [w for w in words if w.lower() not in _STOP]
        alpha = [w for w in content if w[0].isalpha()]
        if len(alpha) == len(content) and alpha and all(w[0].isupper() for w in alpha):
            return 2
    return None


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip()) if s]


def _segments(pages: Sequence[str]) -> List[Tuple[str, int, str]]:
    """Walk the pages and yield (heading_path, page, sentence) in reading order."""
    out: List[Tuple[str, int, str]] = []
    stack: List[Tuple[int, str]] = []  # (level, title)
    for page_no, raw in enumerate(pages, 1):
        buf: List[str] = []

        def flush():
            if buf:
                path = " > ".join(t for _, t in stack)
                for sent in _sentences(" ".join(buf)):
                    out.append((path, page_no, sent))
                buf.clear()

        for line in (raw or "").splitlines():
            lvl = heading_level(line)
            if lvl is None:
                if line.strip():
                    buf.append(line.strip())
                continue
            flush()
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, re.sub(r"\s+", " ", line.strip())))
        flush()
    return out


def structured_chunks(pages: Sequence[str], target_tokens: int = 400, overlap_tokens: int = 60,
                      min_tokens: int = 80) -> List[dict]:
    """Pack sentences into chunks of about `target_tokens`.

    - A new heading starts a new chunk, unless the current chunk is still under half the target size: short
      sections are folded into the next one rather than becoming tiny fragments (so a chunk is labelled
      with the heading it *starts* under).
    - A chunk may continue across a page break; `page` is where it starts, `page_end` where it ends.
    - Consecutive chunks under the same heading overlap by about `overlap_tokens`.
    - A tiny trailing chunk (< min_tokens) is merged into the previous one under the same heading.

    Returns dicts with keys: text, page, page_end, heading."""
    chunks: List[dict] = []
    cur: List[Tuple[int, str]] = []  # (page, sentence)
    cur_tokens = 0
    cur_path = None

    def emit(sentences: List[Tuple[int, str]], path: str):
        chunks.append({"text": " ".join(s for _, s in sentences), "page": sentences[0][0],
                       "page_end": sentences[-1][0], "heading": path})

    def carry_over(sentences: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        keep, tokens = [], 0
        for p, s in reversed(sentences):
            t = estimate_tokens(s)
            if tokens + t > overlap_tokens:
                break
            keep.insert(0, (p, s))
            tokens += t
        return keep

    for path, page, sent in _segments(pages):
        t = estimate_tokens(sent)
        heading_break = path != cur_path and cur_tokens >= max(min_tokens, target_tokens // 2)
        if cur and (heading_break or cur_tokens + t > target_tokens):
            emit(cur, cur_path)
            cur = carry_over(cur) if path == cur_path else []
            cur_tokens = sum(estimate_tokens(s) for _, s in cur)
        if not cur:
            cur_path = path  # a chunk is labelled with the heading it starts under
        cur.append((page, sent))
        cur_tokens += t
    if cur:
        emit(cur, cur_path)

    # merge a tiny chunk into its predecessor when they share a heading
    merged: List[dict] = []
    for c in chunks:
        if (merged and estimate_tokens(c["text"]) < min_tokens and merged[-1]["heading"] == c["heading"]
                and estimate_tokens(merged[-1]["text"]) + estimate_tokens(c["text"]) <= target_tokens * 1.5):
            merged[-1]["text"] += " " + c["text"]
            merged[-1]["page_end"] = c["page_end"]
        else:
            merged.append(c)
    return merged
