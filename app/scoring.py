# app/grader.py

import os
import re
from typing import Dict, Optional
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "tinyllama-qa")  # 👈 updated to fine-tuned model

client = Client(host=OLLAMA_URL)

GRADE_PROMPT = """
You are a strict teacher. Evaluate the student's answer to the following question:

Q: {question}
Correct Answer: {reference}
Student Answer: {student}

Score (0.0 to 1.0): _____
Feedback (short, clear): _____
"""

def grade(reference: str, student: str, question: Optional[str] = None) -> Dict:
    prompt = GRADE_PROMPT.format(
        reference=reference.strip(),
        student=student.strip(),
        question=question.strip() if question else ""
    )
    response = client.generate(model=MODEL, prompt=prompt, stream=False)
    text = response["response"]

    try:
        score_line = [line for line in text.splitlines() if "Score" in line][0]
        feedback_line = [line for line in text.splitlines() if "Feedback" in line][0]
        score = float(score_line.split(":")[-1].strip())
        feedback = feedback_line.split(":", 1)[-1].strip()
        return {"score": score, "feedback": feedback}
    except Exception:
        return {"score": 0.0, "feedback": "Could not parse grading response."}
