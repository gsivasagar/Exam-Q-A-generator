"""PDF text-extraction bake-off: pypdf (current) vs pdfplumber vs pypdfium2 on a folder of PDFs.

Usage: uvx --python 3.13 --with pypdf --with pdfplumber --with pypdfium2 python scripts/extraction_bakeoff.py [pdf_dir]
Reports time, characters, words, glued words (missing spaces) and stray single characters."""
import glob
import hashlib
import re
import sys
import time
from pathlib import Path

import pdfplumber
import pypdf
import pypdfium2

PDF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tmp"
seen=set(); files=[]
for f in sorted(glob.glob(str(PDF_DIR / "*.pdf"))):
    h=hashlib.md5(open(f,"rb").read()).hexdigest()
    if h not in seen: seen.add(h); files.append(f)
def ext_pypdf(p): return [pg.extract_text() or "" for pg in pypdf.PdfReader(p).pages]
def ext_pdfium(p):
    d=pypdfium2.PdfDocument(p); return [pg.get_textpage().get_text_range() for pg in d]
def ext_plumber(p):
    with pdfplumber.open(p) as d: return [pg.extract_text() or "" for pg in d.pages]
def quality(pages):
    t=" ".join(pages); words=re.findall(r"[A-Za-z]{2,}", t)
    # glued words (missing spaces) look like very long alphabetic runs; stray single chars show broken spacing
    long=sum(1 for w in words if len(w)>18); single=len(re.findall(r"(?<!\w)[A-Za-z](?!\w)", t))
    return len(t), len(words), long, single
print(f"{'lib':<11}{'time(s)':>8}{'chars':>9}{'words':>8}{'glued(>18)':>12}{'stray 1-char':>14}")
for name, fn in (("pypdf",ext_pypdf),("pypdfium2",ext_pdfium),("pdfplumber",ext_plumber)):
    t0=time.perf_counter(); tot=[0,0,0,0]
    for f in files:
        q=quality(fn(f)); tot=[a+b for a,b in zip(tot,q)]
    dt=time.perf_counter()-t0
    print(f"{name:<11}{dt:>8.2f}{tot[0]:>9}{tot[1]:>8}{tot[2]:>12}{tot[3]:>14}")
