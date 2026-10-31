from app.qa_generator import ingest_pdf
from tests.conftest import make_pdf_bytes


def test_two_pdfs_same_session_do_not_collide(store):
    a = make_pdf_bytes([f"Alpha document sentence {i}." for i in range(80)])
    b = make_pdf_bytes([f"Beta document sentence {i}." for i in range(80)])
    na = ingest_pdf(None, "s", filename="a.pdf", data=a)
    nb = ingest_pdf(None, "s", filename="b.pdf", data=b)
    chunks = store.get_all_chunks("s")
    assert len(chunks) == na + nb
    assert any("Alpha" in c for c in chunks) and any("Beta" in c for c in chunks)


def test_reingest_same_file_is_idempotent(store, pdf_bytes):
    n = ingest_pdf(None, "s", filename="a.pdf", data=pdf_bytes)
    ingest_pdf(None, "s", filename="a.pdf", data=pdf_bytes)
    assert len(store.get_all_chunks("s")) == n


def test_documents_are_isolated_and_deletable(store, pdf_bytes):
    other = make_pdf_bytes([f"Completely different text {i}." for i in range(80)])
    ingest_pdf(None, "one", filename="a.pdf", data=pdf_bytes)
    ingest_pdf(None, "two", filename="b.pdf", data=other)
    assert not any("Mitochondria" in c for c in store.get_all_chunks("two"))
    store.delete_doc("one")
    assert store.get_all_chunks("one") == []
    assert store.get_all_chunks("two")


def test_similarity_search_scoped(store, pdf_bytes):
    ingest_pdf(None, "one", filename="a.pdf", data=pdf_bytes)
    assert store.similarity_search("mitochondria energy", k=3, doc_id="one")
    assert store.similarity_search("mitochondria energy", k=3, doc_id="nope") == []
