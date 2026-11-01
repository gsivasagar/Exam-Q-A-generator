"""Turn an uploaded file or a web page into chunks. Supports PDF, DOCX, PPTX, Markdown, plain text and URLs.

DOCX and PPTX are read with the standard library (they are zip files of XML), so no extra dependency.
"""
from __future__ import annotations

import io
import ipaddress
import re
import socket
import zipfile
from html.parser import HTMLParser
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from . import config
from .pdf_loader import Chunk, EmptyPDFError, chunks_from_pages, load_pdf

SUPPORTED_EXTENSIONS = ("pdf", "docx", "pptx", "md", "markdown", "txt")


class UploadError(ValueError):
    """The file or URL is not acceptable (unsupported, too large, unsafe, unreadable)."""


# ---------- validation ----------
def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def detect_format(data: bytes, filename: str) -> str:
    """"pdf" | "docx" | "pptx" | "text". Content wins over the file extension."""
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except zipfile.BadZipFile as exc:
            raise UploadError("This file looks like a damaged Office document.") from exc
        if "word/document.xml" in names:
            return "docx"
        if any(n.startswith("ppt/slides/slide") for n in names):
            return "pptx"
        raise UploadError("Unsupported archive. Upload a PDF, DOCX, PPTX, Markdown or text file.")
    if _extension(filename) in ("md", "markdown", "txt"):
        return "text"
    raise UploadError(f"Unsupported file type '{filename}'. Upload a PDF, DOCX, PPTX, Markdown or text file.")


def check_size(size: int, limit_mb: Optional[float] = None) -> None:
    limit = (limit_mb if limit_mb is not None else config.MAX_UPLOAD_MB) * 1024 * 1024
    if size > limit:
        raise UploadError(f"File is too large ({size / 1024 / 1024:.1f} MB; limit "
                          f"{(limit_mb if limit_mb is not None else config.MAX_UPLOAD_MB):g} MB).")


# ---------- readers ----------
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def _safe_read(zf: zipfile.ZipFile, name: str, limit: int = 50 * 1024 * 1024) -> bytes:
    """Read one member, refusing zip bombs (declared size checked, then actual bytes bounded)."""
    info = zf.getinfo(name)
    if info.file_size > limit:
        raise UploadError("Document part is unreasonably large.")
    with zf.open(info) as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise UploadError("Document part is unreasonably large.")
    return data


def read_docx(data: bytes) -> List[str]:
    """One string (headings and paragraphs on separate lines). Word has no fixed pages."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    root = ET.fromstring(_safe_read(zf, "word/document.xml"))
    lines = []
    for p in root.iter(f"{_W}p"):
        text = "".join(t.text or "" for t in p.iter(f"{_W}t")).strip()
        if text:
            lines.append(text)
    return ["\n".join(lines)]


def read_pptx(data: bytes) -> List[str]:
    """One string per slide, in slide order; the title shape comes first."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    def slide_number(name: str) -> int:
        m = re.search(r"(\d+)", name.rsplit("/", 1)[1])
        return int(m.group(1)) if m else 0

    slides = sorted((n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=slide_number)
    out = []
    for name in slides:
        root = ET.fromstring(_safe_read(zf, name))
        title: List[str] = []
        body: List[str] = []
        for sp in root.iter(f"{_P}sp"):
            ph = sp.find(f"{_P}nvSpPr/{_P}nvPr/{_P}ph")
            is_title = ph is not None and ph.get("type") in ("title", "ctrTitle")
            for para in sp.iter(f"{_A}p"):
                text = "".join(t.text or "" for t in para.iter(f"{_A}t")).strip()
                if text:
                    (title if is_title else body).append(text)
        out.append("\n".join(title + body))
    return out


def read_text(data: bytes) -> List[str]:
    # UTF-16 only with a byte-order mark: without one it "decodes" almost any bytes into garbage.
    encodings = ("utf-16", "utf-8-sig") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252")
    text = ""
    for enc in encodings:
        try:
            text = data.decode(enc)
            break
        except UnicodeError:
            continue
    else:
        text = data.decode("latin-1")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # Markdown heading markers: keep the title line
    return [text]


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text)).strip()


# ---------- URLs ----------
def check_url(url: str, resolver: Optional[Callable[[str], List[str]]] = None) -> str:
    """Allow only http(s) URLs whose host resolves to public addresses (blocks localhost, private networks
    and cloud-metadata addresses, so the server cannot be used to reach internal services)."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UploadError("Enter a full web address starting with http:// or https://")
    if parsed.username or parsed.password:
        raise UploadError("Web addresses with embedded credentials are not allowed.")
    host = parsed.hostname

    def resolve(h: str) -> List[str]:
        try:
            return [str(info[4][0]) for info in socket.getaddrinfo(h, None)]
        except socket.gaierror as exc:
            raise UploadError(f"Could not find '{h}'.") from exc

    for addr in (resolver or resolve)(host):
        ip = ipaddress.ip_address(addr.split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified):
            raise UploadError("That address is not allowed (private or local network).")
    return url.strip()


def fetch_url(url: str, fetch: Optional[Callable[[str, int, float], Tuple[bytes, str]]] = None,
              resolver: Optional[Callable[[str], List[str]]] = None) -> Tuple[bytes, str]:
    """Download `url` after safety checks. Returns (bytes, content_type). `fetch` is injectable for tests."""
    check_url(url, resolver)
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    return (fetch or _http_get)(url, limit, config.URL_TIMEOUT)


def _http_get(url: str, limit: int, timeout: float) -> Tuple[bytes, str]:
    import urllib.error
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):  # a redirect could point at an internal address
            return None

    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "exam-qa-generator/0.1"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(limit + 1)
            if len(body) > limit:
                raise UploadError("That page is too large.")
            return body, resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise UploadError("That address redirects elsewhere; paste the final address instead.") from exc
        raise UploadError(f"The site returned an error ({exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UploadError(f"Could not download that page ({exc}).") from exc


# ---------- entry point ----------
def load_document(data: bytes, filename: str, *, ocr=None) -> List[Chunk]:
    """Chunks for an uploaded file. Raises UploadError (unsupported/too large/too many pages) or
    EmptyPDFError (nothing readable)."""
    check_size(len(data))
    kind = detect_format(data, filename)
    if kind == "pdf":
        from pypdf import PdfReader

        try:
            n_pages = len(PdfReader(io.BytesIO(data)).pages)
        except Exception as exc:
            raise UploadError("Could not read this PDF. It may be damaged or password-protected.") from exc
        if n_pages > config.MAX_PDF_PAGES:
            raise UploadError(f"This PDF has {n_pages} pages; the limit is {config.MAX_PDF_PAGES}.")
        return load_pdf(io.BytesIO(data), ocr=ocr)
    try:
        if kind == "docx":
            pages, paged = read_docx(data), False
        elif kind == "pptx":
            pages, paged = read_pptx(data), True
        else:
            pages, paged = read_text(data), False
    except (ET.ParseError, KeyError, zipfile.BadZipFile) as exc:
        raise UploadError(f"Could not read this {kind.upper()} file.") from exc
    chunks = chunks_from_pages(pages, paged=paged)
    if not chunks:
        raise EmptyPDFError("No text found in this file.")
    return chunks


def load_web_page(url: str, **kwargs) -> Tuple[List[Chunk], str]:
    """Download a web page (or PDF) and chunk it. Returns (chunks, display name)."""
    body, ctype = fetch_url(url, **kwargs)
    parsed = urlparse(url)
    name = (parsed.hostname or "web") + (parsed.path.rstrip("/") or "")
    if body[:5] == b"%PDF-" or "pdf" in ctype.lower():
        return load_document(body, name if name.lower().endswith(".pdf") else name + ".pdf"), name
    html = body.decode("utf-8", errors="replace")
    text = html_to_text(html)
    chunks = chunks_from_pages([text], paged=False)
    if not chunks:
        raise EmptyPDFError("No readable text found at that address.")
    return chunks, name
