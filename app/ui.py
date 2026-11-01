# app/ui.py
# Run from the repo root with the package importable: `pip install -e .` (or PYTHONPATH=.),
# then `streamlit run app/ui.py`.
import logging
import uuid

import pandas as pd
import streamlit as st

from app import config
from app.database import load_results, store_results
from app.grader import grade_question
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
TYPE_LABELS = {"Short answer": "short", "Multiple choice": "mcq", "Mixed": "mixed",
               "True / False": "tf", "Fill in the blank": "cloze"}
question_type = TYPE_LABELS[st.radio("Question type", list(TYPE_LABELS), horizontal=True)]
DIFFICULTY_LABELS = {"Any": None, "Easy": "easy", "Medium": "medium", "Hard": "hard"}
difficulty = DIFFICULTY_LABELS[st.radio("Difficulty", list(DIFFICULTY_LABELS), horizontal=True)]


def start_quiz(doc_id, n, topic, qtype, level=None):
    """Generate a quiz from an already-ingested document set and put it in session state."""
    qa_pairs = generate_qa_pairs(doc_id, make_llm(), n=n, topic=topic, question_type=qtype, difficulty=level,
                                 on_route=lambda path: st.session_state.__setitem__("route", path))
    st.session_state.doc_id = doc_id
    st.session_state.quiz_id = str(uuid.uuid4())
    st.session_state.qa_pairs = qa_pairs
    st.session_state.pop("graded", None)
    if len(qa_pairs) < n:
        st.warning(f"Only {len(qa_pairs)} usable questions were generated (requested {n}).")

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
        with st.spinner("Ingesting PDFs and generating questions – please wait …"):
            for f in uploaded_files:
                ingest_pdf(None, doc_id, filename=f.name, data=f.getvalue())
            start_quiz(doc_id, num_q, topic_filter.strip() or None, question_type, difficulty)
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
    ROUTE_NOTES = {"full": "Questions use your whole document.",
                   "topic": "Questions use the parts of your notes that match the topic.",
                   "coverage": "Questions are spread across your whole document."}
    if st.session_state.get("route") in ROUTE_NOTES:
        st.caption(ROUTE_NOTES[st.session_state["route"]])
    with st.form("quiz_form"):
        answers = []
        for idx, qa in enumerate(st.session_state.qa_pairs, 1):
            st.subheader(f"Q{idx}: {qa['question']}")
            key = f"ans_{st.session_state.quiz_id}_{idx}"
            if qa.get("type") in ("mcq", "tf"):
                answers.append(st.radio("Choose one", qa["options"], index=None, key=key) or "")
            else:
                answers.append(st.text_area("Your answer", key=key))
        submitted = st.form_submit_button("Submit answers & grade me")

    if submitted:
        try:
            llm = make_llm()
            graded = []
            bar = st.progress(0.0, text="Grading …")
            total = len(answers)
            for i, (qa, student) in enumerate(zip(st.session_state.qa_pairs, answers), 1):
                res = grade_question(qa, student, llm)
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
            if g.get("explanation"):
                st.markdown(f"*{g['explanation']}*")
            if g.get("source"):
                st.caption(f"📖 Source: {g['source']}" + (f", p.{g['page']}" if g.get("page") else ""))
        st.divider()

    st.write("### Recommended Topics to Review")
    topics = recommend(graded)
    st.write(", ".join(topics) if topics else "Great job – no weak topics detected!")

    if topics and st.button("🎯 Practice my weak topics"):
        try:
            with st.spinner("Building a practice quiz …"):
                start_quiz(st.session_state.doc_id, min(len(graded), 10), topics, question_type, difficulty)
            st.rerun()
        except (LLMError, ValueError) as exc:
            st.error(str(exc))

    st.download_button("📄 Download Quiz Summary as PDF", data=build_pdf(graded),
                       file_name="quiz_results.pdf", mime="application/pdf")

    if st.button("📊 Show Performance Analytics"):
        df = load_results(st.session_state.session_id)
        if df.empty:
            st.info("No history yet.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            st.write("### Performance Over Time (this session)")
            st.line_chart(df.groupby(df["timestamp"].dt.date)["score"].mean())
            st.write("### Average Score by Topic")
            st.bar_chart(df.groupby("topic")["score"].mean().sort_values())
