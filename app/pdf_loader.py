from pathlib import Path
from typing import List
import PyPDF2, re, textwrap

# Simple text cleaner
def _clean(text: str) -> str:
    # collapse whitespace, keep punctuation
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_pdf(path: str | Path, *, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Read a PDF and return cleaned overlapping text chunks."""
    reader = PyPDF2.PdfReader(str(path))
    full_text = " ".join(page.extract_text() or "" for page in reader.pages)
    full_text = _clean(full_text)

    # split into roughly `chunk_size`‑token portions with `overlap` words so context isn’t lost
    words = full_text.split()
    step = chunk_size - overlap
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
    return [textwrap.shorten(chunk, width=chunk_size * 4) for chunk in chunks if chunk]