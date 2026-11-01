# app/ui.py
# Run from the repo root with the package importable: `pip install -e .` (or PYTHONPATH=.),
# then `streamlit run app/ui.py`.
import logging
import re
import uuid

import pandas as pd
import streamlit as st

from app import config
from app.concurrency import effective_workers
from app.database import delete_owner_data, load_results, store_results
from app.exports import exam_pdf, to_anki_tsv, to_csv
from app.grader import grade_all, regrade_failed
from app.llm import LLMError, get_provider
from app.maintenance import cleanup_expired, forget_doc, register_doc
from app.pdf_loader import EmptyPDFError
from app.qa_generator import generate_qa_pairs, ingest_document, ingest_url
from app.quizzes import delete_quiz, list_quizzes, load_quiz, save_quiz
from app.recommendation import recommend
from app.report import build_pdf
from app.srs import count_due, due_questions, record_results
from app.vector_store import delete_doc

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Exam Q&A Generator", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "maintained" not in st.session_state:  # once per browser session: forget old uploaded documents
    st.session_state.maintained = True
    try:
        cleanup_expired()
    except Exception as exc:  # housekeeping must never stop the app
        logging.getLogger(__name__).warning("Cleanup skipped: %s", exc)

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


# Who the saved history belongs to. A profile name lets history survive a browser refresh; it is a label,
# not a login (anyone who types the same name sees the same history).
profile = st.sidebar.text_input("Profile name (optional)",
                                help="Type a name to keep your quizzes and progress between visits.")
owner = re.sub(r"[^a-z0-9_-]+", "-", profile.strip().lower()).strip("-")[:40] or st.session_state.session_id


def begin_attempt(quiz_id, questions, route=None):
    """Put a quiz in front of the student. A fresh attempt id keeps old answers out of the new form."""
    st.session_state.quiz_id = quiz_id
    st.session_state.attempt_id = str(uuid.uuid4())
    st.session_state.qa_pairs = questions
    st.session_state.route = route
    st.session_state.pop("graded", None)




# ───────────── Upload & options ─────────────
uploaded_files = st.file_uploader("Upload one or more study files (PDF, Word, PowerPoint, Markdown, text)",
                                  type=["pdf", "docx", "pptx", "md", "markdown", "txt"], accept_multiple_files=True)
web_url = st.text_input("…or paste a web page address (optional)")
topic_filter = st.text_input("🔍 *Optional topic filter* – only generate questions about this topic")
num_q = st.slider("Number of quiz questions", min_value=5, max_value=25, value=10)
TYPE_LABELS = {"Short answer": "short", "Multiple choice": "mcq", "Mixed": "mixed",
               "True / False": "tf", "Fill in the blank": "cloze"}
question_type = TYPE_LABELS[st.radio("Question type", list(TYPE_LABELS), horizontal=True)]
DIFFICULTY_LABELS = {"Any": None, "Easy": "easy", "Medium": "medium", "Hard": "hard"}
difficulty = DIFFICULTY_LABELS[st.radio("Difficulty", list(DIFFICULTY_LABELS), horizontal=True)]


def start_quiz(doc_id, n, topic, qtype, level=None, title=""):
    """Generate a quiz from an already-ingested document set and put it in session state."""
    bar = st.progress(0.0, text="Generating questions …")
    qa_pairs = generate_qa_pairs(doc_id, make_llm(), n=n, topic=topic, question_type=qtype, difficulty=level,
                                 on_route=lambda path: st.session_state.__setitem__("route", path),
                                 on_progress=lambda done, total: bar.progress(
                                     min(done / total, 1.0), text=f"Generated {done}/{total} questions"))
    bar.empty()
    st.session_state.doc_id = doc_id
    route = st.session_state.get("route")
    begin_attempt(str(uuid.uuid4()), qa_pairs, route)
    topic_text = ", ".join(topic) if isinstance(topic, list) else (topic or "")
    save_quiz(owner, qa_pairs, title=(title or (f"Practice: {topic_text}" if topic_text else "Quiz"))[:80],
              question_type=qtype, difficulty=level,
              quiz_id=st.session_state.quiz_id)
    if len(qa_pairs) < n:
        st.warning(f"Only {len(qa_pairs)} usable questions were generated (requested {n}).")

# ───────────── Create quiz ─────────────
if (uploaded_files or web_url.strip()) and st.button("Create Quiz"):
    # Fresh document set per quiz so earlier uploads never leak into this one.
    doc_id = str(uuid.uuid4())
    old = st.session_state.get("doc_id")
    if old:
        delete_doc(old)
        forget_doc(old)
    for key in ("qa_pairs", "graded"):
        st.session_state.pop(key, None)
    try:
        register_doc(doc_id)
        with st.spinner("Ingesting your files and generating questions – please wait …"):
            for f in uploaded_files or []:
                ingest_document(None, doc_id, filename=f.name, data=f.getvalue())
            if web_url.strip():
                ingest_url(web_url.strip(), doc_id)
            names = ", ".join([f.name for f in uploaded_files or []] + ([web_url.strip()] if web_url.strip() else []))
            start_quiz(doc_id, num_q, topic_filter.strip() or None, question_type, difficulty, title=names)
        st.success("Quiz ready! Scroll down to begin ⬇️")
    except (EmptyPDFError, LLMError, ValueError) as exc:
        delete_doc(doc_id)
        forget_doc(doc_id)
        st.error(str(exc))

# ───────────── Take the quiz ─────────────
if "qa_pairs" in st.session_state:
    st.header("📝 Take the quiz")
    ROUTE_NOTES = {"full": "Questions use your whole document.",
                   "topic": "Questions use the parts of your notes that match the topic.",
                   "coverage": "Questions are spread across your whole document."}
    if st.session_state.get("route") in ROUTE_NOTES:
        st.caption(ROUTE_NOTES[st.session_state["route"]])
    with st.expander("⬇️ Export this quiz"):
        qs = st.session_state.qa_pairs
        e1, e2, e3, e4 = st.columns(4)
        e1.download_button("Printable exam (PDF)", exam_pdf(qs), "exam.pdf", "application/pdf")
        e2.download_button("Exam + answer key (PDF)", exam_pdf(qs, with_key=True), "exam_with_key.pdf",
                           "application/pdf")
        e3.download_button("Anki cards (.txt)", to_anki_tsv(qs), "anki_cards.txt", "text/plain")
        e4.download_button("Spreadsheet (.csv)", to_csv(qs).encode("utf-8-sig"), "quiz.csv", "text/csv")

    with st.form("quiz_form"):
        answers = []
        for idx, qa in enumerate(st.session_state.qa_pairs, 1):
            st.subheader(f"Q{idx}: {qa['question']}")
            key = f"ans_{st.session_state.attempt_id}_{idx}"
            if qa.get("type") in ("mcq", "tf"):
                answers.append(st.radio("Choose one", qa["options"], index=None, key=key) or "")
            else:
                answers.append(st.text_area("Your answer", key=key))
        submitted = st.form_submit_button("Submit answers & grade me")

    if submitted:
        try:
            llm = make_llm()
            bar = st.progress(0.0, text="Grading …")
            results = grade_all(st.session_state.qa_pairs, answers, llm, workers=effective_workers(llm),
                                on_progress=lambda done, total: bar.progress(done / total,
                                                                             text=f"Graded {done}/{total}"))
            graded = [{**qa, "student": student, **res}
                      for qa, student, res in zip(st.session_state.qa_pairs, answers, results)]
            bar.empty()
            store_results(graded, owner, st.session_state.quiz_id)
            record_results(owner, graded)
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
            if g.get("key_points") and g.get("covered") and len(g["covered"]) == len(g["key_points"]):
                st.markdown("**Key points**")
                for point, ok in zip(g["key_points"], g["covered"]):
                    st.markdown(f"{'✅' if ok else '❌'} {point}")
            elif g.get("key_points"):
                st.markdown("**Key points**\n" + "\n".join(f"- {p}" for p in g["key_points"]))
            if g.get("explanation"):
                st.markdown(f"*{g['explanation']}*")
            if g.get("source"):
                st.caption(f"📖 Source: {g['source']}" + (f", p.{g['page']}" if g.get("page") else ""))
        st.divider()

    if any(g.get("error") for g in graded) and st.button("🔁 Retry ungraded answers"):
        try:
            llm = make_llm()
            before = list(graded)
            graded = regrade_failed(graded, llm, workers=effective_workers(llm))
            newly = [g for g, old in zip(graded, before) if old.get("error") and not g.get("error")]
            if newly:
                store_results(newly, owner, st.session_state.quiz_id)
                record_results(owner, newly)
            st.session_state.graded = graded
            st.rerun()
        except LLMError as exc:
            st.error(str(exc))

    st.write("### Recommended Topics to Review")
    topics = recommend(graded)
    st.write(", ".join(topics) if topics else "Great job – no weak topics detected!")

    if topics and st.session_state.get("doc_id") and st.button("🎯 Practice my weak topics"):
        try:
            with st.spinner("Building a practice quiz …"):
                start_quiz(st.session_state.doc_id, min(len(graded), 10), topics, question_type, difficulty)
            st.rerun()
        except (LLMError, ValueError) as exc:
            st.error(str(exc))

    st.download_button("📄 Download Quiz Summary as PDF", data=build_pdf(graded),
                       file_name="quiz_results.pdf", mime="application/pdf")

    if st.button("📊 Show Performance Analytics"):
        df = load_results(owner)
        if df.empty:
            st.info("No history yet.")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            st.write("### Performance Over Time")
            st.line_chart(df.groupby(df["timestamp"].dt.date)["score"].mean())
            st.write("### Average Score by Topic")
            st.bar_chart(df.groupby("topic")["score"].mean().sort_values())

# ───────────── Sidebar: history (drawn last so it reflects what just happened) ─────────────
with st.sidebar.expander("📚 My quizzes"):
    saved = list_quizzes(owner, limit=10)
    if not saved:
        st.caption("Quizzes you create are saved here for retaking.")
    for q in saved:
        score = "" if q["avg_score"] is None else f" · avg {q['avg_score']:.0%}"
        st.caption(f"{q['title']} ({q['n']} Qs{score})")
        c1, c2 = st.columns(2)
        if c1.button("Retake", key=f"retake_{q['id']}"):
            full = load_quiz(owner, q["id"])
            if full:
                begin_attempt(full["id"], full["questions"])
                st.rerun()
        if c2.button("Delete", key=f"del_{q['id']}"):
            delete_quiz(owner, q["id"])
            st.rerun()

n_due = count_due(owner)
if st.sidebar.button(f"🧠 Review due questions ({n_due})", disabled=n_due == 0,
                     help="Questions you missed (or answered a while ago) that are due again."):
    begin_attempt(str(uuid.uuid4()), due_questions(owner, limit=10))
    st.rerun()

with st.sidebar.expander("🗑️ My data"):
    st.caption("Deletes your saved quizzes, results and review schedule.")
    if st.checkbox("Yes, delete my data", key="confirm_delete") and st.button("Delete now"):
        counts = delete_owner_data(owner)
        st.success(f"Deleted {sum(counts.values())} item(s).")
