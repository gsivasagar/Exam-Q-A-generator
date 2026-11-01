"""Optional OCR for scanned PDFs (no text layer).

Needs two optional packages plus the Tesseract program:
    pip install pypdfium2 pytesseract      and      brew install tesseract   (or apt install tesseract-ocr)
OCR=auto (default) uses them only when a PDF has no extractable text and they are installed;
OCR=off never runs OCR.
"""
from __future__ import annotations

import io
from typing import List

from . import config


class OCRUnavailable(RuntimeError):
    pass


def available() -> bool:
    try:
        import pypdfium2  # noqa: F401
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_pdf(data: bytes, *, max_pages: int = 0) -> List[str]:
    """Return one recognised text string per page, or [] when OCR is disabled or unavailable
    (the caller then reports a scanned-PDF error)."""
    if config.OCR_MODE == "off" or not available():
        return []
    import pypdfium2
    import pytesseract

    limit = max_pages or config.MAX_PDF_PAGES
    pdf = pypdfium2.PdfDocument(io.BytesIO(data))
    texts = []
    for i in range(min(len(pdf), limit)):
        image = pdf[i].render(scale=2).to_pil()
        texts.append(pytesseract.image_to_string(image))
    return texts
