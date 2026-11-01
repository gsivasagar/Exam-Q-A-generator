"""Typed shapes shared across the pipeline. Plain dicts at runtime (they flow into SQLite, PDF and UI code)."""
from __future__ import annotations

from typing import List, Optional, TypedDict


class Question(TypedDict, total=False):
    question: str
    answer: str
    type: str                 # "short" | "mcq"
    topic: str
    source: Optional[str]     # file name the question was based on
    page: Optional[int]       # 1-based page in that file
    options: List[str]        # mcq only: answer + distractors, shuffled
