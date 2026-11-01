"""Parallel LLM calls, waves, grading in parallel, and the response cache."""
import json
import threading
import time

import pytest

from app import config
from app.concurrency import effective_workers, run_parallel
from app.grader import grade_all
from app.llm import LLMError
from app.llm_cache import CachedProvider
from app.planning import plan_wave
from app.qa_generator import generate_qa_pairs, ingest_pdf
from tests.conftest import FakeLLM, make_pdf_bytes


class TestRunParallel:
    def test_results_in_input_order(self):
        def slow_first(x):
            time.sleep(0.05 if x == 0 else 0)
            return x * 2
        assert run_parallel(slow_first, [0, 1, 2, 3], workers=4) == [0, 2, 4, 6]

    def test_actually_concurrent(self):
        start = time.perf_counter()
        run_parallel(lambda _: time.sleep(0.1), range(4), workers=4)
        assert time.perf_counter() - start < 0.3  # sequential would be ~0.4

    def test_one_worker_runs_inline_in_caller_thread(self):
        seen = []
        run_parallel(lambda _: seen.append(threading.current_thread()), [1, 2], workers=1)
        assert set(seen) == {threading.current_thread()}

    def test_progress_called_from_calling_thread(self):
        threads, counts = set(), []
        run_parallel(lambda x: x, range(5), workers=3,
                     on_done=lambda d, t: (threads.add(threading.current_thread()), counts.append((d, t))))
        assert threads == {threading.current_thread()} and counts[-1] == (5, 5)

    def test_first_error_reraised_after_all_finish(self):
        done = []

        def fn(x):
            time.sleep(0.02)
            done.append(x)
            if x == 1:
                raise LLMError("boom")
            return x
        with pytest.raises(LLMError):
            run_parallel(fn, [0, 1, 2, 3], workers=4)
        assert sorted(done) == [0, 1, 2, 3]

    def test_empty(self):
        assert run_parallel(lambda x: x, [], workers=4) == []


class TestWorkers:
    def test_provider_limit_wins_and_is_capped(self, monkeypatch):
        class P:
            max_concurrency = 4
        assert effective_workers(P()) == 4
        P.max_concurrency = 999
        assert effective_workers(P()) == 16

    def test_default_is_sequential(self):
        assert effective_workers(FakeLLM()) == config.LLM_CONCURRENCY == 1

    def test_provider_defaults(self):
        from app.llm import GeminiProvider, OllamaProvider
        assert OllamaProvider(model="m", host="http://x:1").max_concurrency == 1
        assert GeminiProvider("k").max_concurrency == 4


class TestPlanWave:
    def test_size_one_is_the_sequential_plan(self):
        assert plan_wave({"short": 10}, [], set(), per_call=5, size=1) == [("short", 5)]

    def test_wave_splits_shortfall_across_requests(self):
        assert plan_wave({"short": 10}, [], set(), 5, size=4) == [("short", 5), ("short", 5)]

    def test_mixed_types_stay_balanced(self):
        wave = plan_wave({"mcq": 5, "short": 5}, [], set(), 5, size=2)
        assert sorted(t for t, _ in wave) == ["mcq", "short"]

    def test_nothing_needed(self):
        assert plan_wave({"short": 1}, [{"type": "short"}], set(), 5, size=3) == []

    def test_failed_types_skipped(self):
        assert plan_wave({"mcq": 5, "short": 5}, [], {"mcq"}, 5, size=3)[0][0] == "short"


class SlowLLM(FakeLLM):
    def __init__(self, delay=0.15):
        super().__init__()
        self.delay = delay
        self.lock = threading.Lock()
        self.max_concurrent = 0
        self._active = 0
        self.max_concurrency = 1

    def generate(self, prompt, *, schema=None, **_):
        with self.lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls += 1
            call = self.calls
        time.sleep(self.delay)
        with self.lock:
            self._active -= 1
        if schema and "questions" in schema.get("properties", {}):
            return json.dumps({"questions": [{"question": f"Unique question {call}-{i}?", "answer": f"a{i}",
                                              "topic": "T", "chunk": 1} for i in range(5)]})
        return '{"score": 0.5, "feedback": "ok"}'


def notes():
    return make_pdf_bytes([f"Mitochondria produce energy number {i} for the cell." for i in range(20)])


class TestGenerationInParallel:
    def test_parallel_wave_is_faster_and_correct(self, store):
        ingest_pdf(None, "d", filename="a.pdf", data=notes())
        seq = SlowLLM()
        t0 = time.perf_counter()
        a = generate_qa_pairs("d", seq, n=10)
        t_seq = time.perf_counter() - t0
        par = SlowLLM()
        par.max_concurrency = 4
        t0 = time.perf_counter()
        b = generate_qa_pairs("d", par, n=10)
        t_par = time.perf_counter() - t0
        assert len(a) == len(b) == 10
        assert par.max_concurrent >= 2 and seq.max_concurrent == 1
        assert t_par < t_seq * 0.75

    def test_sequential_never_overlaps(self, store):
        ingest_pdf(None, "d", filename="a.pdf", data=notes())
        llm = SlowLLM(0.02)
        generate_qa_pairs("d", llm, n=10)
        assert llm.max_concurrent == 1

    def test_progress_callback(self, store):
        ingest_pdf(None, "d", filename="a.pdf", data=notes())
        seen = []
        generate_qa_pairs("d", SlowLLM(0), n=10, on_progress=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1] == (10, 10) and all(t == 10 for _, t in seen)
        assert [d for d, _ in seen] == sorted(d for d, _ in seen)


class TestGradeAll:
    QAS = [{"type": "short", "question": f"q{i}", "answer": "a"} for i in range(4)] + [
        {"type": "mcq", "question": "m", "answer": "x", "options": ["x", "y"]}]

    def test_in_order_and_parallel(self):
        llm = SlowLLM(0.1)
        start = time.perf_counter()
        out = grade_all(self.QAS, ["s"] * 4 + ["x"], llm, workers=4)
        assert time.perf_counter() - start < 0.3
        assert [r["score"] for r in out] == [0.5] * 4 + [1.0]

    def test_progress_and_sequential_equal(self):
        counts = []
        a = grade_all(self.QAS, ["s"] * 4 + ["x"], SlowLLM(0), workers=1, on_progress=lambda d, t: counts.append(d))
        b = grade_all(self.QAS, ["s"] * 4 + ["x"], SlowLLM(0), workers=3)
        assert a == b and counts == [1, 2, 3, 4, 5]

    def test_errors_become_ungraded_not_crashes(self):
        out = grade_all(self.QAS[:2], ["s", "s"], FakeLLM([LLMError("down"), LLMError("down")]), workers=2)
        assert all(r["score"] is None for r in out)


class TestCache:
    def test_hit_after_miss_and_distinct_keys(self):
        inner = FakeLLM(['{"a": 1}', '{"a": 2}', '{"a": 3}'])
        inner.model = "m"
        c = CachedProvider(inner)
        assert c.generate("p") == '{"a": 1}'
        assert c.generate("p") == '{"a": 1}'                       # served from cache
        assert c.generate("p", temperature=0) == '{"a": 2}'         # different temperature = different key
        assert c.generate("q", schema={"x": 1}) == '{"a": 3}'
        assert (c.hits, c.misses, inner.calls) == (1, 3, 3)

    def test_persists_on_disk(self, tmp_path):
        inner = FakeLLM(["first"])
        CachedProvider(inner, str(tmp_path)).generate("p")
        again = CachedProvider(FakeLLM([LLMError("must not be called")]), str(tmp_path))
        assert again.generate("p") == "first" and again.hits == 1

    def test_failures_not_cached(self):
        inner = FakeLLM([LLMError("down"), "ok"])
        c = CachedProvider(inner)
        with pytest.raises(LLMError):
            c.generate("p")
        assert c.generate("p") == "ok"

    def test_delegates_attributes(self):
        inner = FakeLLM()
        inner.context_chars = 123
        assert CachedProvider(inner).context_chars == 123

    def test_corrupt_cache_file_is_ignored(self, tmp_path):
        inner = FakeLLM(["fresh"])
        c = CachedProvider(inner, str(tmp_path))
        (tmp_path / f"{c._key('p', None, None)}.json").write_text("{not json")
        assert c.generate("p") == "fresh"

    def test_get_provider_wraps_when_configured(self, monkeypatch, tmp_path):
        from app import llm
        monkeypatch.setattr(config, "LLM_CACHE_DIR", str(tmp_path))
        assert isinstance(llm.get_provider("Ollama", ollama_model="m"), CachedProvider)
        monkeypatch.setattr(config, "LLM_CACHE_DIR", None)
        assert not isinstance(llm.get_provider("Ollama", ollama_model="m"), CachedProvider)
