"""PDF export of quiz results."""
from __future__ import annotations

from typing import Dict, List

from fpdf import FPDF


def _safe(text, maxlen: int = 600) -> str:
    """Latin-1-safe, clipped, with very long unbroken words wrapped."""
    s = "N/A" if text is None else " ".join(str(text).split())
    s = " ".join(s[i:i + 60] for i in range(0, len(s), 60))
    if len(s) > maxlen:
        s = s[:maxlen] + "..."
    return s.encode("latin-1", "replace").decode("latin-1")


def build_pdf(results: List[Dict]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Exam Quiz Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=12)
    for idx, g in enumerate(results, 1):
        score = g.get("score")
        score_txt = "not graded" if score is None else f"{score:.0%}"
        for line in (f"Q{idx}: {g.get('question')}", f"Your Score: {score_txt}",
                     f"Your Answer: {g.get('student')}", f"Correct Answer: {g.get('answer')}",
                     f"Feedback: {g.get('feedback')}",
                     *( [f"Source: {g['source']}" + (f", p.{g['page']}" if g.get('page') else "")]
                        if g.get("source") else [] )):
            pdf.multi_cell(0, 8, _safe(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
    return bytes(pdf.output())
