# app/grader.py

import os
import re
from typing import Dict, Optional
from dotenv import load_dotenv
from ollama import Client
import google.generativeai as genai

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

def grade(
    reference: str, 
    student: str, 
    question: Optional[str] = None,
    provider: str = "Ollama",
    gemini_api_key: Optional[str] = None,
    ollama_model: str = "tinyllama-qa"
) -> Dict:
    prompt = GRADE_PROMPT.format(
        reference=reference.strip(),
        student=student.strip(),
        question=question.strip() if question else ""
    )

    if provider == "Gemini":
        if not gemini_api_key:
            return {"score": 0.0, "feedback": "Gemini API key is required when provider is Gemini"}
        genai.configure(api_key=gemini_api_key)
        # Using a model tailored for evaluation tasks or flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = response.text
    else:
        response = client.generate(model=ollama_model, prompt=prompt, stream=False)
        text = response["response"]

    try:
        score_line = [line for line in text.splitlines() if "Score" in line][0]
        feedback_line = [line for line in text.splitlines() if "Feedback" in line][0]
        score = float(score_line.split(":")[-1].strip())
        feedback = feedback_line.split(":", 1)[-1].strip()
        return {"score": score, "feedback": feedback}
    except Exception:
        return {"score": 0.0, "feedback": "Could not parse grading response."}
