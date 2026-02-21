"""
Generate quiz question‑answer pairs using a fine‑tuned Ollama model.
Supports an optional `topic` argument to bias questions toward a theme.
"""

from __future__ import annotations
import os, json, random, re
from typing import List, Dict, Optional

from ollama import Client
import google.generativeai as genai
from .vector_store import add_chunks, similarity_search
from .pdf_loader import load_pdf

# ──────────────────────────────────────────────────────────────────────────────
# 1. Ollama client & global prompt setup
# ──────────────────────────────────────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")  # Your fine-tuned model name
_client = Client(host=OLLAMA_URL)

_BASE_SYSTEM_PROMPT = """
You are a professional exam tutor. Based on the textbook context, generate ONLY a JSON list.

STRICTLY follow this format:
[
  {"question": "What is ...?", "answer": "...", "topic": "TopicName"},
  ...
]

Do NOT include numbering, explanations, markdown, or any text outside the list. Output MUST start with `[`.
Use ONLY the context below.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 2. Ingest and store PDF chunks
# ──────────────────────────────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str, doc_id: str) -> int:
    """
    Load a PDF file, split it into text chunks, and store in the vector DB.
    """
    chunks = load_pdf(pdf_path)
    add_chunks(chunks, doc_id)
    return len(chunks)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Generate question–answer pairs using context chunks
# ──────────────────────────────────────────────────────────────────────────────
def generate_qa_pairs(
    doc_id: str,
    n: int = 10,
    topic: Optional[str] = None,
    provider: str = "Ollama",
    gemini_api_key: Optional[str] = None,
    ollama_model: str = "gemma3:latest"
) -> List[Dict[str, str]]:
    query = topic if topic else "general"
    all_chunks = similarity_search(query=query, k=50, doc_id=doc_id)

    if not all_chunks:
        raise ValueError(f"No chunks found for document: {doc_id}")

    selected_chunks = random.sample(all_chunks, k=min(len(all_chunks), 8))
    context = "\n".join(selected_chunks)[:2000]

    instruction = (
        f"Generate {n} question‑answer pairs focused on the topic '{topic}'."
        if topic else f"Generate {n} general exam-style question‑answer pairs."
    )

    prompt = (
        f"{_BASE_SYSTEM_PROMPT.strip()}\n\n"
        f"{instruction.strip()}\n\n"
        f"Context:\n{context.strip()}"
    )

    if provider == "Gemini":
        if not gemini_api_key:
            raise ValueError("Gemini API key is required when provider is Gemini")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        response_text = response.text
    else:
        response = _client.generate(model=ollama_model, prompt=prompt)
        response_text = response["response"]

    print("📦 MODEL RAW RESPONSE:\n", response_text)  # ✅ Right here

    try:
        return json.loads(_extract_first_json_array(response_text))
    except Exception as exc:
        raise ValueError("Could not parse model response as JSON list") from exc


# ──────────────────────────────────────────────────────────────────────────────
# 4. Extract JSON array from LLM response text
# ──────────────────────────────────────────────────────────────────────────────
def _extract_first_json_array(text: str) -> str:
    """
    Extract the first valid JSON array from a string. Cleans bad characters.
    """
    import re

    print("\n🪵 RAW RESPONSE FOR EXTRACTION:\n", text)  # Debug print

    # Find the first list starting with [ and ending with ]
    match = re.search(r"\[\s*{[\s\S]+?}\s*\]", text)
    if not match:
        raise ValueError("❌ No valid JSON array found in model response.")

    json_text = match.group(0)

    # Replace curly quotes with standard ones
    json_text = json_text.replace("“", '"').replace("”", '"')
    json_text = json_text.replace("‘", "'").replace("’", "'")

    return json_text
