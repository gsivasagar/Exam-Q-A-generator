"""Export a quiz: Anki import file, CSV, and a printable exam PDF (optionally with an answer key)."""
from __future__ import annotations

import csv
import html
import io
from typing import Dict, List

from fpdf import FPDF

from .report import _safe

LETTERS = "ABCDEFGH"


def _oneline(text) -> str:
    return " ".join(str(text or "").split())


def to_anki_tsv(questions: List[Dict]) -> str:
    """Tab-separated 'front<TAB>back<TAB>tags' for Anki's File > Import. HTML-escaped; line breaks as <br>."""
    rows = []
    for q in questions:
        front = html.escape(_oneline(q["question"]))
        if q.get("options"):
            front += "<br>" + "<br>".join(f"{LETTERS[i]}) {html.escape(_oneline(o))}"
                                          for i, o in enumerate(q["options"]))
        back = html.escape(_oneline(q["answer"]))
        if q.get("explanation"):
            back += "<br><i>" + html.escape(_oneline(q["explanation"])) + "</i>"
        if q.get("source"):
            cite = _oneline(q["source"]) + (f", p.{q['page']}" if q.get("page") else "")
            back += "<br><small>" + html.escape(cite) + "</small>"
        tag = _oneline(q.get("topic") or "general").replace(" ", "_")
        rows.append(f"{front}\t{back}\t{tag}")
    return "\n".join(rows) + ("\n" if rows else "")


def to_csv(questions: List[Dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question", "answer", "type", "options", "topic", "difficulty", "source", "page", "explanation",
                "key_points"])
    for q in questions:
        w.writerow([q.get("question", ""), q.get("answer", ""), q.get("type", ""), " | ".join(q.get("options", [])),
                    q.get("topic", ""), q.get("difficulty", ""), q.get("source") or "", q.get("page") or "",
                    q.get("explanation", ""), " | ".join(q.get("key_points", []))])
    return buf.getvalue()


def exam_pdf(questions: List[Dict], *, title: str = "Practice Exam", with_key: bool = False) -> bytes:
    """A printable exam: numbered questions, lettered options, blank lines for written answers.
    With with_key=True an answer key follows on a new page."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe(title, 80), new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "Name: ______________________    Date: ______________", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for idx, q in enumerate(questions, 1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, _safe(f"{idx}. {q['question']}", 700), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        if q.get("options"):
            for i, opt in enumerate(q["options"]):
                pdf.multi_cell(0, 6, _safe(f"    {LETTERS[i]}) {opt}", 300), new_x="LMARGIN", new_y="NEXT")
        elif q.get("type") == "cloze":
            pass
        else:
            for _ in range(3):
                pdf.cell(0, 8, "_" * 85, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    if with_key:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Answer key", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        for idx, q in enumerate(questions, 1):
            ans = q["answer"]
            if q.get("options") and ans in q["options"]:
                ans = f"{LETTERS[q['options'].index(ans)]}) {ans}"
            line = f"{idx}. {ans}"
            if q.get("source"):
                line += f"   [{q['source']}" + (f", p.{q['page']}" if q.get("page") else "") + "]"
            pdf.multi_cell(0, 6, _safe(line, 500), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
