"""Chunking stats over a folder of PDFs (default: ./tmp). Usage: python scripts/chunk_stats.py [dir]"""
import sys, glob, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.pdf_loader import load_pdf, EmptyPDFError
from app import qa_generator as q
rows=[]; chars=[]; words=[]; fit=[]
for f in sorted(glob.glob(str((Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp") / "*.pdf"))):
    try: ch = load_pdf(f)
    except EmptyPDFError: rows.append((f,0)); continue
    except Exception as e: print("ERR", f, e); continue
    rows.append((f,len(ch)))
    chars += [len(c.text) for c in ch]; words += [len(c.text.split()) for c in ch]
    # simulate prompt packing: how many of the first 4 chunks fit in MAX_CONTEXT_CHARS
    for i in range(0, len(ch)-3, 4):
        ctx=0; n=0
        for j,c in enumerate(ch[i:i+4],1):
            p=len(f"[{j}] {c.text}\n\n")
            if ctx+p>q.MAX_CONTEXT_CHARS: break
            ctx+=p; n+=1
        fit.append(n)
print("pdfs", len(rows), "empty", sum(1 for _,n in rows if n==0), "total chunks", sum(n for _,n in rows))
print("chunk chars mean/median/p95/max", round(st.mean(chars)), st.median(chars), sorted(chars)[int(.95*len(chars))], max(chars))
print("chunk words mean/median", round(st.mean(words)), st.median(words))
print("chunks per pdf median", st.median([n for _,n in rows]))
from collections import Counter
print("chunks fitting per 4-chunk batch:", dict(Counter(fit)), "mean", round(st.mean(fit),2))
