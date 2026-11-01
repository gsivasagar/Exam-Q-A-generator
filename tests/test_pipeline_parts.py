"""Unit tests for the pieces split out of qa_generator."""
import random

from app.dedupe import dedupe
from app.planning import BATCH_SIZE, max_rounds, next_request, trim_to_quota, type_quota
from app.prompting import MAX_CONTEXT_CHARS, build_context, build_prompt, schema_for
from app.selection import CHUNKS_PER_BATCH, BatchCycler, normalize_topics


class TestPlanning:
    def test_quota(self):
        assert type_quota("short", 10) == {"short": 10}
        assert type_quota("mcq", 7) == {"mcq": 7}
        assert type_quota("mixed", 11) == {"mcq": 5, "short": 6}

    def test_max_rounds_grows_with_n(self):
        assert max_rounds(5) < max_rounds(25)

    def test_next_request_picks_largest_shortfall(self):
        quota = {"mcq": 5, "short": 6}
        pairs = [{"type": "short"}] * 4
        assert next_request(quota, pairs, set()) == ("mcq", 5)

    def test_next_request_skips_failed_and_finished(self):
        quota = {"mcq": 2, "short": 2}
        pairs = [{"type": "short"}] * 2
        assert next_request(quota, pairs, {"mcq"}) is None
        assert next_request(quota, pairs, set()) == ("mcq", 2)

    def test_trim_to_quota(self):
        pairs = [{"type": "short", "i": i} for i in range(5)] + [{"type": "mcq", "i": i} for i in range(5)]
        out = trim_to_quota(pairs, {"mcq": 2, "short": 3}, "mixed")
        assert sorted((p["type"] for p in out)) == ["mcq"] * 2 + ["short"] * 3
        assert len(trim_to_quota(pairs, {"short": 3}, "short")) == 3

    def test_batch_size_constant(self):
        assert BATCH_SIZE == 5


class TestSelection:
    def test_normalize_topics(self):
        assert normalize_topics(None) == []
        assert normalize_topics("  ") == []
        assert normalize_topics("loops") == ["loops"]
        assert normalize_topics(["a", " ", 3, "b"]) == ["a", "b"]

    def test_cycler_wraps_and_covers_all_chunks(self):
        random.seed(0)
        hits = [{"text": f"c{i}"} for i in range(6)]
        cyc = BatchCycler(list(hits))
        seen = set()
        for _ in range(3):
            batch = cyc.next_batch()
            assert len(batch) == CHUNKS_PER_BATCH
            seen |= {h["text"] for h in batch}
        assert seen == {h["text"] for h in hits}

    def test_cycler_small_document(self):
        cyc = BatchCycler([{"text": "only"}, {"text": "two"}])
        assert len(cyc.next_batch()) == 2


class TestPrompting:
    def test_context_is_numbered_and_capped(self):
        hits = [{"text": "x" * 2500} for _ in range(5)]
        ctx = build_context(hits)
        assert ctx.startswith("[1] ") and len(ctx) <= MAX_CONTEXT_CHARS
        assert "[3]" not in ctx

    def test_prompt_mentions_topic_and_type(self):
        p = build_prompt([{"text": "t"}], 3, "Recursion", "mcq")
        assert "focused on: Recursion" in p and "distractors" in p
        assert "distractors" not in build_prompt([{"text": "t"}], 3, None, "short")

    def test_schema_requires_distractors_only_for_mcq(self):
        mcq = schema_for("mcq")["properties"]["questions"]["items"]
        short = schema_for("short")["properties"]["questions"]["items"]
        assert "distractors" in mcq["required"] and "distractors" not in short["required"]


def test_dedupe_ignores_case_and_punctuation():
    pairs = [{"question": "What is X?"}, {"question": "what is x"}, {"question": "What is Y?"}]
    assert [p["question"] for p in dedupe(pairs)] == ["What is X?", "What is Y?"]
