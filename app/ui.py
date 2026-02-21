# app/ui.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # ensure “app” package is importable

import os
import uuid
import json
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
from fpdf import FPDF

from app.pdf_loader import load_pdf           # if you want to inspect raw chunks
from app.qa_generator import ingest_pdf, generate_qa_pairs
from app.scoring import grade
from app.recommendation import recommend
from app.database import init_db, store_results  # we'll create this file in a moment

from fpdf import FPDF
import textwrap

st.set_page_config(page_title="Exam Q&A Generator", page_icon="📚", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# 1. SESSION & PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("📚 Exam Q&A Generator")

st.sidebar.title("⚙️ Configuration")
provider = st.sidebar.radio("Select LLM Provider", ["Ollama", "Gemini"])
gemini_api_key = ""
if provider == "Gemini":
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_api_key:
            st.sidebar.warning("Please enter your Gemini API key.")
elif provider == "Ollama":
    ollama_model = st.sidebar.text_input("Ollama Model", value="gemma3:latest")
    with st.sidebar.expander("ℹ️ Ollama Setup Instructions"):
        st.markdown(
            f"1. Download and install [Ollama](https://ollama.com).\n"
            f"2. Make sure the Ollama app is running.\n"
            f"3. Pull the required model by running this command in your terminal:\n"
            f"```bash\n"
            f"ollama pull {ollama_model}\n"
            f"```"
        )
# ──────────────────────────────────────────────────────────────────────────────
# 2. PDF UPLOAD & QUIZ GENERATION OPTIONS
# ──────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more study PDFs",
    type="pdf",
    accept_multiple_files=True,
)

topic_filter = st.text_input(
    "🔍 *Optional topic filter* – generate questions only about this topic (leave blank for full coverage)"
)

num_q = st.slider("Number of quiz questions", min_value=5, max_value=25, value=10)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CREATE QUIZ
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_files and st.button("Create Quiz"):
    with st.spinner("Ingesting PDFs and generating questions – please wait …"):
        # Store PDFs to tmp/ and ingest each one
        tmp_dir = Path("tmp")
        tmp_dir.mkdir(exist_ok=True)

        for file in uploaded_files:
            pdf_path = tmp_dir / f"{uuid.uuid4()}.pdf"
            pdf_path.write_bytes(file.read())
            ingest_pdf(str(pdf_path), doc_id=st.session_state.session_id)

        if provider == "Gemini" and not gemini_api_key:
            st.error("Please provide a Gemini API Key in the sidebar.")
            st.stop()

        # Generate quiz questions
        qa_kwargs = {
            "doc_id": st.session_state.session_id,
            "n": num_q,
            "topic": topic_filter.strip() if topic_filter else None,
            "provider": provider,
            "gemini_api_key": gemini_api_key
        }
        if provider == "Ollama":
             qa_kwargs["ollama_model"] = ollama_model
             
        qa_pairs = generate_qa_pairs(**qa_kwargs)

        st.session_state.qa_pairs = qa_pairs
        st.success("Quiz ready! Scroll down to begin ⬇️")

# ──────────────────────────────────────────────────────────────────────────────
# 4. QUIZ UI
# ──────────────────────────────────────────────────────────────────────────────
if "qa_pairs" in st.session_state:
    st.header("📝 Take the quiz")
    answers = []

    for idx, qa in enumerate(st.session_state.qa_pairs, 1):
        st.subheader(f"Q{idx}: {qa['question']}")
        ans = st.text_area("Your answer", key=f"ans_{idx}")
        answers.append(ans)

    # ────────── Grade Button ──────────
    if st.button("Submit answers & grade me"):
        graded = []
        with st.spinner("Grading …"):
            for qa, student in zip(st.session_state.qa_pairs, answers):
                if not student.strip():
                    graded.append(
                        {**qa, "student": student, "score": 0.0, "feedback": "No answer submitted."}
                    )
                else:
                    grade_kwargs = {
                        "reference": qa["answer"],
                        "student": student,
                        "question": qa["question"],
                        "provider": provider,
                        "gemini_api_key": gemini_api_key
                    }
                    if provider == "Ollama":
                        grade_kwargs["ollama_model"] = ollama_model
                        
                    res = grade(**grade_kwargs)
                    graded.append({**qa, "student": student, **res})

        # Persist and store
        init_db()
        store_results(graded)

        st.session_state.graded = graded
        st.success("Results ready!")

# ──────────────────────────────────────────────────────────────────────────────
# 5. RESULTS & FEEDBACK
# ──────────────────────────────────────────────────────────────────────────────
if "graded" in st.session_state:
    graded = st.session_state.graded
    scores = [g["score"] for g in graded]
    num_correct = sum(s >= 0.6 for s in scores)

    st.header("Your Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Score", f"{sum(scores)/len(scores):.2%}")
    with col2:
        st.metric("Questions Correct (≥0.6)", f"{num_correct} / {len(graded)}")

    st.write("### Detailed Feedback")
    for idx, g in enumerate(graded, 1):
        st.markdown(f"**Q{idx}:** {g['question']}")
        st.markdown(f"**Your score:** {g['score']:.0%}")
        st.markdown(f"**Feedback:** {g['feedback']}")
        with st.expander("Correct answer"):
            st.markdown(g["answer"])
        st.divider()

    # Recommended topics
    st.write("### Recommended Topics to Review")
    topics = recommend(graded)
    if topics:
        st.write(", ".join(topics))
    else:
        st.write("Great job – no weak topics detected!")

    # ────────── Export as PDF ──────────
    def generate_pdf(results):
        """
        Robust PDF generator that never crashes, even with long or odd text.
        """
        from fpdf import FPDF

        def latin1(text, maxlen=300):
            """Return a latin‑1–safe, clipped string."""
            if text is None:
                return "N/A"
            s = str(text).replace("\n", " ").replace("\r", " ").strip()
            # Break long unspaced words every 50 chars
            s = " ".join([s[i:i + 50] for i in range(0, len(s), 50)])
            if len(s) > maxlen:
                s = s[:maxlen] + "..."
            # Replace characters outside Latin‑1 range
            return s.encode("latin1", "replace").decode("latin1")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, latin1("Exam Quiz Summary"), ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", size=12)

        full_width = pdf.w - pdf.l_margin - pdf.r_margin  # safe width for all cells

        for idx, g in enumerate(results, 1):
            pdf.multi_cell(full_width, 8, latin1(f"Q{idx}: {g.get('question')}"))
            # Use cell (single‑line) for score to avoid wrapping issues
            pdf.cell(full_width, 8, latin1(f"Your Score: {g.get('score', 0):.0%}"), ln=True)
            pdf.multi_cell(full_width, 8, latin1(f"Your Answer: {g.get('student')}"))
            pdf.multi_cell(full_width, 8, latin1(f"Correct Answer: {g.get('answer')}"))
            pdf.multi_cell(full_width, 8, latin1(f"Feedback: {g.get('feedback')}"))
            pdf.ln(4)

        return bytes(pdf.output(dest="S"))  

    pdf_bytes = generate_pdf(graded)
    st.download_button(
        "📄 Download Quiz Summary as PDF",
        data=pdf_bytes,
        file_name="quiz_results.pdf",
        mime="application/pdf",
    )

    # ────────── Analytics ──────────
    if st.button("📊 Show Performance Analytics"):
        conn = sqlite3.connect("quiz_results.db")
        df = pd.read_sql_query("SELECT * FROM results", conn)
        conn.close()

        st.write("### Performance Over Time")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        daily_avg = df.groupby("date")["score"].mean()
        st.line_chart(daily_avg)

        st.write("### Average Score by Question")
        q_avg = df.groupby("question")["score"].mean().sort_values()
        st.bar_chart(q_avg)

# ──────────────────────────────────────────────────────────────────────────────
# 6. END OF FILE
# ──────────────────────────────────────────────────────────────────────────────
