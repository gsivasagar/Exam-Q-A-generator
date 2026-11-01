"""Structure-aware chunking: headings, page spanning, token sizing, parent expansion."""
import pytest

from app import config, vector_store
from app.chunking import estimate_tokens, heading_level, structured_chunks
from app.pdf_loader import Chunk, load_pdf
from tests.conftest import make_pdf_bytes


class TestHeadingDetection:
    @pytest.mark.parametrize("line,level", [
        ("1 Introduction", 1), ("2.1 Cell Biology", 2), ("3.2.1 Krebs Cycle Steps", 3),
        ("Chapter 4: Energy", 1), ("PHOTOSYNTHESIS", 1), ("Cell Structure and Function", 2),
    ])
    def test_headings(self, line, level):
        assert heading_level(line) == level

    @pytest.mark.parametrize("line", [
        "", "This is an ordinary sentence about cells.", "the mitochondria is an organelle",
        "1 2 3", "3.14", "Water boils at 100 degrees, which is well known.", "x" * 200,
    ])
    def test_not_headings(self, line):
        assert heading_level(line) is None


def words(n, tag="w"):
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


class TestStructuredChunks:
    def test_new_heading_starts_new_chunk_with_path(self):
        pages = ["1 Cells\n" + words(30) + "\n1.1 Nucleus\n" + words(30) + "\n2 Energy\n" + words(30)]
        chunks = structured_chunks(pages, target_tokens=40, min_tokens=1)  # sections >= half the target
        assert [c["heading"] for c in chunks] == ["1 Cells", "1 Cells > 1.1 Nucleus", "2 Energy"]

    def test_short_sections_fold_into_the_next_instead_of_fragmenting(self):
        pages = ["1 A\nShort one.\n2 B\nShort two.\n3 C\nShort three."]
        (c,) = structured_chunks(pages, target_tokens=400, min_tokens=1)
        assert c["heading"] == "1 A" and "Short three." in c["text"]

    def test_chunk_spans_pages_and_reports_range(self):
        pages = ["1 Topic\nFirst sentence on page one.", "Second sentence on page two."]
        (c,) = structured_chunks(pages, min_tokens=1)
        assert (c["page"], c["page_end"]) == (1, 2) and "page two" in c["text"]

    def test_sized_in_tokens_with_overlap(self):
        sentence = ("alpha " * 20).strip() + "."          # ~30 tokens
        pages = ["1 Big\n" + " ".join([sentence] * 40)]
        chunks = structured_chunks(pages, target_tokens=200, overlap_tokens=60, min_tokens=1)
        assert len(chunks) > 3
        assert all(estimate_tokens(c["text"]) <= 200 + 40 for c in chunks)
        # consecutive chunks under one heading share their boundary sentence(s)
        assert chunks[1]["text"].split(". ")[0] in chunks[0]["text"]

    def test_tiny_tail_merged_into_previous(self):
        pages = ["1 Topic\n" + " ".join([("beta " * 20).strip() + "."] * 8) + " Tail."]
        chunks = structured_chunks(pages, target_tokens=100, overlap_tokens=0, min_tokens=40)
        assert not chunks[-1]["text"] == "Tail."

    def test_empty_input(self):
        assert structured_chunks([""]) == []
        assert structured_chunks([]) == []

    def test_no_headings_still_chunks(self):
        chunks = structured_chunks([" ".join([("gamma " * 20).strip() + "."] * 30)], target_tokens=150)
        assert len(chunks) >= 3 and all(c["heading"] == "" for c in chunks)


class TestLoadPdfModes:
    def test_structured_mode_returns_headings(self, tmp_path):
        data = make_pdf_bytes(["Some plain text about mitochondria and energy production in cells."] * 5)
        import io
        chunks = load_pdf(io.BytesIO(data), mode="structured")
        assert chunks and all(isinstance(c, Chunk) for c in chunks)

    def test_legacy_is_default(self, monkeypatch):
        import io
        data = make_pdf_bytes(["Plain sentence number one is here."] * 3)
        monkeypatch.setattr(config, "CHUNKING", "legacy")
        assert all(c.heading == "" and c.page_end is None for c in load_pdf(io.BytesIO(data)))


class TestParentExpansion:
    def _chunks(self):
        return [Chunk("alpha one.", 1, "H"), Chunk("beta two.", 1, "H"), Chunk("gamma three.", 2, "H"),
                Chunk("other topic.", 2, "Z")]

    def test_parent_text_joins_neighbours_under_same_heading(self):
        parents = vector_store._parent_texts(self._chunks())
        assert parents[1] == "alpha one. beta two. gamma three."
        assert parents[3] == "other topic."  # different heading: no merge

    def test_search_returns_parent_when_enabled(self, store, monkeypatch):
        monkeypatch.setattr(config, "PARENT_EXPANSION", True)
        store.add_chunks(self._chunks(), "d", source="s.pdf", file_id="f")
        hits = store.similarity_search("beta two", k=3, doc_id="d")
        assert any(h["text"] == "alpha one. beta two. gamma three." for h in hits)
        assert len({h["text"] for h in hits}) == len(hits)  # de-duplicated

    def test_search_returns_child_when_disabled(self, store, monkeypatch):
        monkeypatch.setattr(config, "PARENT_EXPANSION", False)
        store.add_chunks(self._chunks(), "d", source="s.pdf", file_id="f")
        hits = store.similarity_search("beta two", k=4, doc_id="d")
        assert "beta two." in {h["text"] for h in hits}

    def test_heading_and_page_end_round_trip(self, store):
        store.add_chunks([Chunk("text here.", 3, "1 Intro", 4)], "d", source="s.pdf", file_id="f")
        (h,) = store.get_all_chunks("d")
        assert (h["heading"], h["page"], h["page_end"]) == ("1 Intro", 3, 4)
