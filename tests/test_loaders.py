"""Ingestion of PDF/DOCX/PPTX/Markdown/text/URLs, upload limits, URL safety and OCR fallback."""
import io
import zipfile

import pytest
from fpdf import FPDF

from app import config
from app.loaders import (UploadError, check_size, check_url, detect_format, fetch_url, html_to_text,
                         load_document, load_web_page, read_docx, read_pptx, read_text)
from app.pdf_loader import EmptyPDFError, load_pdf
from app.qa_generator import ingest_document, ingest_pdf, ingest_url
from tests.conftest import FakeLLM, make_pdf_bytes

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def make_docx(paragraphs):
    body = "".join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def make_pptx(slides):
    """slides: list of (title, [body lines])."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        for i, (title, body) in enumerate(slides, 1):
            body_xml = "".join(f"<a:p><a:r><a:t>{b}</a:t></a:r></a:p>" for b in body)
            xml = (f'<?xml version="1.0"?><p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>'
                   f'<p:sp><p:nvSpPr><p:cNvPr id="1" name="b"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                   f'<p:txBody>{body_xml}</p:txBody></p:sp>'
                   f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="t"/><p:cNvSpPr/><p:nvPr><p:ph type="title"/></p:nvPr>'
                   f'</p:nvSpPr><p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>'
                   f'</p:spTree></p:cSld></p:sld>')
            z.writestr(f"ppt/slides/slide{i}.xml", xml)
    return buf.getvalue()


def blank_pdf():
    pdf = FPDF()
    pdf.add_page()  # a page with no text: a stand-in for a scanned document
    return bytes(pdf.output())


LONG = ["Mitochondria produce energy in the cell. " * 6] * 3


class TestFormatDetection:
    def test_by_content(self):
        assert detect_format(make_pdf_bytes(["x."]), "weird.bin") == "pdf"
        assert detect_format(make_docx(["a"]), "x.dat") == "docx"
        assert detect_format(make_pptx([("T", ["b"])]), "x.dat") == "pptx"

    def test_text_by_extension(self):
        assert detect_format(b"hello", "notes.md") == "text"
        assert detect_format(b"hello", "NOTES.TXT") == "text"

    @pytest.mark.parametrize("name", ["a.exe", "noext", "a.png"])
    def test_unsupported(self, name):
        with pytest.raises(UploadError, match="Unsupported"):
            detect_format(b"MZ\x90\x00", name)

    def test_unknown_zip_and_corrupt_zip(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a.txt", "x")
        with pytest.raises(UploadError):
            detect_format(buf.getvalue(), "a.zip")
        with pytest.raises(UploadError, match="damaged"):
            detect_format(b"PK\x03\x04garbage", "a.docx")


class TestReaders:
    def test_docx_paragraphs(self):
        assert read_docx(make_docx(["Title", "First para.", "", "Second."])) == ["Title\nFirst para.\nSecond."]

    def test_pptx_slide_order_and_title_first(self):
        pages = read_pptx(make_pptx([("Intro", ["a1", "a2"]), ("Next", ["b1"])]))
        assert pages == ["Intro\na1\na2", "Next\nb1"]

    def test_text_encodings_and_markdown_markers(self):
        assert read_text("# Heading\nbody".encode()) == ["Heading\nbody"]
        assert read_text("café".encode("latin-1")) == ["café"]
        assert read_text("caf\u00e9 ok".encode("utf-16")) == ["café ok"]   # with BOM
        assert read_text("﻿hello".encode("utf-8")) == ["hello"]

    def test_html_to_text_skips_scripts_and_nav(self):
        html = ("<html><head><title>x</title><style>p{}</style></head><body><nav>menu</nav>"
                "<h1>Cells</h1><p>Cells divide.</p><script>evil()</script><p>Mitosis.</p></body></html>")
        text = html_to_text(html)
        assert "Cells divide." in text and "Mitosis." in text
        assert "evil" not in text and "menu" not in text and "p{}" not in text

    def test_zip_member_size_guard(self):
        from app.loaders import _safe_read
        z = zipfile.ZipFile(io.BytesIO(make_docx(["x" * 500])))
        with pytest.raises(UploadError, match="unreasonably large"):
            _safe_read(z, "word/document.xml", limit=100)


class TestLoadDocument:
    def test_pdf_docx_pptx_text_all_chunk(self):
        pdf = load_document(make_pdf_bytes(["Cells divide by mitosis."] * 30), "a.pdf")
        docx = load_document(make_docx(LONG), "a.docx")
        slides = [("Cells", ["Cells divide. " * 30]), ("DNA", ["DNA stores genes. " * 30])]
        pptx = load_document(make_pptx(slides), "a.pptx")
        txt = load_document(("Plain text sentence. " * 60).encode(), "a.txt")
        assert all(chunks for chunks in (pdf, docx, pptx, txt))
        assert pdf[0].page == 1
        assert all(c.page == 0 for c in docx + txt)            # no real pages: page 0 = "no page"
        assert [c.page for c in pptx][0] == 1 and max(c.page for c in pptx) == 2   # slides act as pages

    def test_empty_files_report_clearly(self):
        with pytest.raises(EmptyPDFError):
            load_document(make_docx([""]), "a.docx")
        with pytest.raises(EmptyPDFError):
            load_document(b"   \n ", "a.txt")

    def test_size_limit(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)
        with pytest.raises(UploadError, match="too large"):
            load_document(b"x" * 5000, "a.txt")
        check_size(10, 1)  # within limit: no error

    def test_page_limit(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_PDF_PAGES", 2)
        pdf = FPDF()
        for _ in range(3):
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)
            pdf.cell(0, 10, "Some text on the page.")
        with pytest.raises(UploadError, match="3 pages; the limit is 2"):
            load_document(bytes(pdf.output()), "a.pdf")

    def test_damaged_pdf(self):
        with pytest.raises(UploadError, match="Could not read this PDF"):
            load_document(b"%PDF-1.4 this is not really a pdf", "a.pdf")

    def test_damaged_docx_xml(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("word/document.xml", "<not closed")
        with pytest.raises(UploadError, match="Could not read this DOCX"):
            load_document(buf.getvalue(), "a.docx")


class TestOCR:
    def test_scanned_pdf_without_ocr_is_rejected_clearly(self, monkeypatch):
        monkeypatch.setattr(config, "OCR_MODE", "off")
        with pytest.raises(EmptyPDFError, match="scanned"):
            load_pdf(io.BytesIO(blank_pdf()))

    def test_ocr_callable_is_used_when_there_is_no_text_layer(self):
        chunks = load_pdf(io.BytesIO(blank_pdf()), ocr=lambda raw: ["Recognised text about cells. " * 5])
        assert chunks and "Recognised text" in chunks[0].text and chunks[0].page == 1

    def test_ocr_not_called_when_text_exists(self):
        def boom(raw):
            raise AssertionError("OCR should not run")
        assert load_pdf(io.BytesIO(make_pdf_bytes(["Real text here."] * 5)), ocr=boom)

    def test_ocr_that_finds_nothing_still_errors(self):
        with pytest.raises(EmptyPDFError):
            load_pdf(io.BytesIO(blank_pdf()), ocr=lambda raw: ["", " "])

    def test_ocr_module_disabled_or_missing_returns_nothing(self, monkeypatch):
        from app import ocr
        monkeypatch.setattr(config, "OCR_MODE", "off")
        assert ocr.ocr_pdf(b"x") == []
        monkeypatch.setattr(config, "OCR_MODE", "auto")
        monkeypatch.setattr(ocr, "available", lambda: False)
        assert ocr.ocr_pdf(b"x") == []


class TestUrlSafety:
    PUBLIC = staticmethod(lambda host: ["93.184.216.34"])

    def test_public_https_allowed(self):
        assert check_url("https://example.com/notes", self.PUBLIC) == "https://example.com/notes"

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)",
                                     "example.com", "http://", "https://user:pw@example.com/"])
    def test_bad_schemes_and_forms(self, url):
        with pytest.raises(UploadError):
            check_url(url, self.PUBLIC)

    @pytest.mark.parametrize("addr", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.9", "169.254.169.254",
                                      "::1", "0.0.0.0", "fe80::1", "fd00::1"])
    def test_private_and_local_addresses_blocked(self, addr):
        with pytest.raises(UploadError, match="not allowed"):
            check_url("http://internal.example/x", lambda h: [addr])

    def test_any_private_answer_blocks(self):
        with pytest.raises(UploadError):
            check_url("http://rebind.example/", lambda h: ["93.184.216.34", "127.0.0.1"])

    def test_unresolvable_host(self):
        with pytest.raises(UploadError, match="Could not find"):
            check_url("http://no-such-host.invalid/")

    def test_fetch_checks_before_downloading(self):
        called = []
        with pytest.raises(UploadError):
            fetch_url("http://localhost/x", fetch=lambda *a: called.append(1), resolver=lambda h: ["127.0.0.1"])
        assert not called


class TestWebPages:
    def fake(self, body, ctype="text/html"):
        return lambda url, limit, timeout: (body, ctype)

    def test_html_page(self):
        html = ("<html><body><h1>Cells</h1>" + "<p>Cells divide by mitosis. </p>" * 30 + "</body></html>").encode()
        chunks, name = load_web_page("https://example.com/bio", fetch=self.fake(html),
                                     resolver=lambda h: ["93.184.216.34"])
        assert name == "example.com/bio" and chunks and all(c.page == 0 for c in chunks)

    def test_pdf_url(self):
        pdf = make_pdf_bytes(["Cells divide by mitosis."] * 30)
        chunks, name = load_web_page("https://example.com/n.pdf", fetch=self.fake(pdf, "application/pdf"),
                                     resolver=lambda h: ["93.184.216.34"])
        assert chunks and chunks[0].page == 1

    def test_empty_page(self):
        with pytest.raises(EmptyPDFError):
            load_web_page("https://example.com/x", fetch=self.fake(b"<html><script>x()</script></html>"),
                          resolver=lambda h: ["93.184.216.34"])


class TestIngest:
    def test_ingest_document_all_formats_and_alias(self, store):
        assert ingest_pdf is ingest_document
        n1 = ingest_document(None, "d", filename="a.docx", data=make_docx(LONG))
        n2 = ingest_document(None, "d", filename="b.md", data=("# Cells\n" + "Cells divide by mitosis. " * 40).encode())
        assert n1 > 0 and n2 > 0
        sources = {h["source"] for h in store.get_all_chunks("d")}
        assert sources == {"a.docx", "b.md"}

    def test_ingest_document_is_idempotent(self, store):
        data = make_docx(LONG)
        a = ingest_document(None, "d", filename="a.docx", data=data)
        ingest_document(None, "d", filename="a.docx", data=data)
        assert len(store.get_all_chunks("d")) == a

    def test_ingest_url(self, store, monkeypatch):
        import app.loaders as loaders
        html = ("<p>Cells divide by mitosis. </p>" * 40).encode()
        monkeypatch.setattr(loaders, "_http_get", lambda url, limit, timeout: (html, "text/html"))
        monkeypatch.setattr(loaders.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
        assert ingest_url("https://example.com/bio", "d") > 0
        assert store.get_all_chunks("d")[0]["source"] == "example.com/bio"

    def test_quiz_from_docx_end_to_end(self, store):
        from app.qa_generator import generate_qa_pairs
        import json

        class Cites(FakeLLM):
            def generate(self, prompt, *, schema=None, **_):
                self.calls += 1
                return json.dumps({"questions": [{"question": f"Q{self.calls}-{i}?", "answer": "a", "topic": "T",
                                                  "chunk": 1} for i in range(5)]})

        ingest_document(None, "d", filename="notes.docx", data=make_docx(LONG))
        out = generate_qa_pairs("d", Cites(), n=5)
        assert len(out) == 5 and all(q["source"] == "notes.docx" and q["page"] == 0 for q in out)
