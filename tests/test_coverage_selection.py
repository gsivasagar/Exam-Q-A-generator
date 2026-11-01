"""Coverage-aware chunk selection: clustering, round-robin batches, strategy switch."""
import random

import numpy as np
import pytest

from app.clustering import kmeans, mean_pairwise_similarity
from app.selection import CHUNKS_PER_BATCH, BatchCycler, CoverageCycler, make_cycler


def blobs(n_per=8, centers=((1, 0, 0), (0, 1, 0), (0, 0, 1)), noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    vecs, truth = [], []
    for ci, c in enumerate(centers):
        for _ in range(n_per):
            vecs.append(np.array(c, dtype=float) + rng.normal(0, noise, len(c)))
            truth.append(ci)
    return vecs, truth


class TestKMeans:
    def test_separates_obvious_groups(self):
        vecs, truth = blobs()
        labels = kmeans(vecs, 3)
        for ci in range(3):
            assert len({labels[i] for i, t in enumerate(truth) if t == ci}) == 1
        assert len(set(labels)) == 3

    def test_deterministic(self):
        vecs, _ = blobs()
        assert kmeans(vecs, 3) == kmeans(vecs, 3)

    def test_k_larger_than_n_and_identical_points(self):
        assert len(set(kmeans([[1.0, 0.0]] * 3, 5))) >= 1
        assert len(kmeans([[1.0, 0.0], [0.0, 1.0]], 10)) == 2

    def test_mean_pairwise_similarity(self):
        assert mean_pairwise_similarity([[1, 0]]) == 0.0
        assert mean_pairwise_similarity([[1, 0], [1, 0]]) == pytest.approx(1.0)
        assert mean_pairwise_similarity([[1, 0], [0, 1]]) == pytest.approx(0.0)


def hits_from(vecs, truth):
    return [{"text": f"c{i}", "source": "s.pdf", "page": 1 + t, "embedding": v, "truth": t}
            for i, (v, t) in enumerate(zip(vecs, truth))]


class TestCoverageCycler:
    def test_first_batch_spans_different_groups(self):
        random.seed(1)
        vecs, truth = blobs(n_per=8, centers=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
        cyc = CoverageCycler(hits_from(vecs, truth))
        batch = cyc.next_batch()
        assert len(batch) == CHUNKS_PER_BATCH
        assert len({h["truth"] for h in batch}) == 4  # one from each group, never 4 from one

    def test_visits_every_chunk_before_repeating(self):
        random.seed(2)
        vecs, truth = blobs(n_per=4)
        hits = hits_from(vecs, truth)  # 12 chunks
        cyc = CoverageCycler(hits)
        seen = []
        for _ in range(3):
            seen += [h["text"] for h in cyc.next_batch()]
        assert sorted(seen) == sorted(h["text"] for h in hits)

    def test_no_duplicate_inside_a_batch_across_refill(self):
        random.seed(3)
        vecs, truth = blobs(n_per=2, centers=((1, 0, 0), (0, 1, 0), (0, 0, 1)))  # 6 chunks, batches of 4
        cyc = CoverageCycler(hits_from(vecs, truth))
        for _ in range(10):
            texts = [h["text"] for h in cyc.next_batch()]
            assert len(texts) == len(set(texts)) == 4

    def test_beats_random_on_diversity(self):
        vecs, truth = blobs(n_per=10)
        hits = hits_from(vecs, truth)
        cov_groups, rnd_groups = [], []
        for seed in range(30):
            random.seed(seed)
            cov_groups.append(len({h["truth"] for h in CoverageCycler([dict(h) for h in hits]).next_batch()}))
            rnd_groups.append(len({h["truth"] for h in BatchCycler([dict(h) for h in hits]).next_batch()}))
        assert np.mean(cov_groups) > np.mean(rnd_groups)


class TestStrategySwitch:
    def test_coverage_used_when_embeddings_present(self):
        vecs, truth = blobs()
        assert isinstance(make_cycler(hits_from(vecs, truth), "coverage"), CoverageCycler)

    def test_random_strategy(self):
        vecs, truth = blobs()
        assert isinstance(make_cycler(hits_from(vecs, truth), "random"), BatchCycler)

    def test_falls_back_without_embeddings_or_for_tiny_docs(self):
        assert isinstance(make_cycler([{"text": str(i)} for i in range(10)], "coverage"), BatchCycler)
        vecs, truth = blobs(n_per=1, centers=((1, 0, 0), (0, 1, 0)))
        assert isinstance(make_cycler(hits_from(vecs, truth), "coverage"), BatchCycler)

    def test_unknown_strategy_rejected(self):
        with pytest.raises(ValueError):
            make_cycler([], "magic")

    def test_default_comes_from_config(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "SELECTION_STRATEGY", "random")
        vecs, truth = blobs()
        assert isinstance(make_cycler(hits_from(vecs, truth)), BatchCycler)


def test_generate_qa_pairs_accepts_strategy(tmp_path):
    from app.qa_generator import generate_qa_pairs, ingest_pdf
    from tests.conftest import FakeLLM, make_pdf_bytes

    sentences = [f"Topic {i} fact number {j} is important." for i in range(3) for j in range(60)]
    ingest_pdf(None, "cov", filename="a.pdf", data=make_pdf_bytes(sentences))
    for strategy in ("coverage", "random"):
        out = generate_qa_pairs("cov", FakeLLM(), n=5, strategy=strategy)
        assert len(out) == 5
