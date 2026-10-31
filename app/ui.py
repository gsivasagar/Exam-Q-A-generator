# app/ui.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # make "app" importable

import logging
import uuid

import streamlit as st

from app import config
from app.database import load_results, store_results
from app.grader import grade
from app.llm import LLMError, get_provider
from app.pdf_loader import EmptyPDFError
from app.qa_generator import generate_qa_pairs, ingest_pdf
from app.recommendation import recommend
from app.report import build_pdf
from app.vector_store import delete_doc

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Exam Q&A Generator", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("📚 Exam Q&A Generator")

# ───────────── Sidebar ─────────────
st.sidebar.title("⚙️ Configuration")
provider = st.sidebar.radio("Select LLM Provider", ["Ollama", "Gemini"])
gemini_api_key, ollama_model = "", config.OLLAMA_MODEL
if provider == "Gemini":
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password") or config.GEMINI_API_KEY
    if not gemini_api_key:
        st.sidebar.warning("Please enter your Gemini API key.")
else:
    ollama_model = st.sidebar.text_input("Ollama Model", value=config.OLLAMA_MODEL)
    with st.sidebar.expander("ℹ️ Ollama Setup Instructions"):
        st.markdown(
            "1. Download and install [Ollama](https://ollama.com).\n"
            "2. Make sure the Ollama app is running.\n"
            f"3. Pull the model:\n```bash\nollama pull {ollama_model}\n```"
        )


def make_llm():
    return get_provider(provider, gemini_api_key=gemini_api_key, ollama_model=ollama_model)


# ───────────── Upload & options ─────────────
uploaded_files = st.file_uploader("Upload one or more study PDFs", type="pdf", accept_multiple_files=True)
topic_filter = st.text_input("🔍 *Optional topic filter* – only generate questions about this topic")
num_q = st.slider("Number of quiz questions", min_value=5, max_value=25, value=10)

# ───────────── Create quiz ─────────────
if uploaded_files and st.button("Create Quiz"):
    # Fresh document set per quiz so earlier uploads never leak into this one.
    doc_id = str(uuid.uuid4())
    old = st.session_state.get("doc_id")
    if old:
        delete_doc(old)
    for key in ("qa_pairs", "graded"):
        st.session_state.pop(key, None)
    try:
        llm = make_llm()
        with st.spinner("Ingesting PDFs and generating questions – please wait …"):
            for f in uploaded_files:
                ingest_pdf(None, doc_id, filename=f.name, data=f.getvalue())
            qa_pairs = generate_qa_pairs(doc_id, llm, n=num_q, topic=topic_filter.strip() or None)
        st.session_state.doc_id = doc_id
        st.session_state.quiz_id = str(uuid.uuid4())
        st.session_state.qa_pairs = qa_pairs
        if len(qa_pairs) < num_q:
            st.warning(f"Only {len(qa_pairs)} usable questions were generated (requested {num_q}).")
        st.success("Quiz ready! Scroll down to begin ⬇️")
    except EmptyPDFError as exc:
        delete_doc(doc_id)
        st.error(str(exc))
    except (LLMError, ValueError) as exc:
        delete_doc(doc_id)
        st.error(str(exc))

# ───────────── Take the quiz ─────────────
if "qa_pairs" in st.session_state:
    st.header("📝 Take the quiz")
    with st.form("quiz_form"):
        answers = []
        for idx, qa in enumerate(st.session_state.qa_pairs, 1):
            st.subheader(f"Q{idx}: {qa['question']}")
            answers.append(st.text_area("Your answer", key=f"ans_{idx}"))
        submitted = st.form_submit_button("Submit answers & grade me")

    if submitted:
        try:
            llm = make_llm()
            graded = []
            bar = st.progress(0.0, text="Grading …")
            total = len(answers)
            for i, (qa, student) in enumerate(zip(st.session_state.qa_pairs, answers), 1):
                if not student.strip():
                    res = {"score": 0.0, "feedback": "No answer submitted."}
                else:
                    res = grade(qa["answer"], student, llm, question=qa["question"])
                graded.append({**qa, "student": student, **res})
                bar.progress(i / total, text=f"Graded {i}/{total}")
            bar.empty()
            store_results(graded, st.session_state.session_id, st.session_state.quiz_id)
            st.session_state.graded = graded
            if any(g.get("error") for g in graded):
                st.warning("Some answers could not be graded automatically and are excluded from your score.")
            st.success("Results ready!")
        except LLMError as exc:
            st.error(str(exc))

# ───────────── Results ─────────────
if "graded" in st.session_state:
    graded = st.session_state.graded
    scored = [g for g in graded if g["score"] is not None]
    st.header("Your Results")
    if scored:
        avg = sum(g["score"] for g in scored) / len(scored)
        correct = sum(g["score"] >= config.PASS_THRESHOLD for g in scored)
        c1, c2 = st.columns(2)
        c1.metric("Average Score", f"{avg:.0%}")
        c2.metric(f"Questions Correct (≥{config.PASS_THRESHOLD})", f"{correct} / {len(scored)}")

    st.write("### Detailed Feedback")
    for idx, g in enumerate(graded, 1):
        st.markdown(f"**Q{idx}:** {g['question']}")
        st.markdown("**Your score:** " + ("not graded" if g["score"] is None else f"{g['score']:.0%}"))
        st.markdown(f"**Feedback:** {g['feedback']}")
        with st.expander("Correct answer"):
            st.markdown(g["answer"])
        st.divider()

    st.write("### Recommended Topics to Review")
    topics = recommend(graded)
    st.write(", ".join(topics) if topics else "Great job – no weak topics detected!")

    st.download_button("📄 Download Quiz Summary as PDF", data=build_pdf(graded),
                       file_name="quiz_results.pdf", mime="application/pdf")

    if st.button("📊 Show Performance Analytics"):
        df = load_results(st.session_state.session_id)
        if df.empty:
            st.info("No history yet.")
        else:
            df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")
            st.write("### Performance Over Time (this session)")
            st.line_chart(df.groupby(df["timestamp"].dt.date)["score"].mean())
            st.write("### Average Score by Topic")
            st.bar_chart(df.groupby("topic")["score"].mean().sort_values())
