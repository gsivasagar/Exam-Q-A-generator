# Exam Q&A Generator with Ollama 📚

An interactive tool that generates educational quizzes from your study PDF notes using local LLMs via [Ollama](https://ollama.com).

## ✨ Features
*   **PDF Ingestion**: Upload your study materials directly.
*   **AI Question Generation**: Automatically creates quiz questions based on your content.
*   **Question Types**: Short answer, multiple choice, or a mix of both.
*   **Two LLM Providers**: Local Ollama (default) or Google Gemini (API key required; document text is sent to Google).
*   **Interactive Quiz**: Take the quiz directly in your browser.
*   **Auto-Grading**: Multiple choice is graded instantly; short answers are graded by the LLM.
*   **Source Citations**: Each question shows the file and page it came from.
*   **Weak-Topic Practice**: Regenerate a quiz focused on the topics you missed.
*   **Performance Tracking**: Charts of your scores over time and by topic.
*   **PDF Export**: Download a summary of your results.

## 🚀 Quick Start (Docker)

The easiest way to run the application is with Docker.

### Prerequisites
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
*   [Ollama](https://ollama.com/) installed and running locally.
*   Pull a model (e.g., `ollama pull gemma3` or your preferred model).

### Running the App
1.  **Clone the repository** (if you haven't already).
2.  **Ensure Ollama is running**.
3.  **Start the container**:
    ```bash
    docker-compose up --build
    ```
4.  **Open in Browser**:
    Go to [http://localhost:8501](http://localhost:8501)

> **Tip**: The database and vector store are persisted in `./data` so you won't lose data when restarting the container.

## 🛠 Manual Installation

If you prefer running without Docker:

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install the project** (this also makes the `app` package importable):
    ```bash
    pip install -e .
    ```
3.  **Run the App** (from the repository root):
    ```bash
    streamlit run app/ui.py
    ```

## 🔧 Configuration
*   **Model**: Defaults to `gemma3:latest`. Change `OLLAMA_MODEL` in `docker-compose.yml` or `.env` to use a different model.
*   **Ollama URL**: Defaults to `http://localhost:11434`.
*   **Embedding model**: `EMBEDDING_MODEL` is `default` (built-in MiniLM, no setup) or `ollama:<model>`, e.g. `ollama:bge-m3` or `ollama:nomic-embed-text` (run `ollama pull <model>` first). Each embedder uses its own vector store, so switching never mixes vectors.
*   **Chunk selection**: `SELECTION_STRATEGY=coverage` (default) or `random`.
*   **Grading**: `GRADING_MODE=rubric` (default) scores short answers by the key points they cover, computed in code; `holistic` asks the model for a 0-1 score. Blank / "I don't know" and exact-match answers are scored without calling the model.
*   **Speed**: independent LLM calls can run in parallel. Gemini defaults to 4 at once (`GEMINI_CONCURRENCY`); Ollama defaults to 1 (`OLLAMA_CONCURRENCY`; raise it together with Ollama's `OLLAMA_NUM_PARALLEL`). `LLM_CACHE_DIR` caches identical requests on disk (useful for development and evaluation).

## 🧪 Development
```bash
pip install -e . -r requirements-dev.txt
pytest
ruff check app tests        # lint (pip install ruff, or: uvx ruff check app tests)
```
Optional: `pre-commit install` runs ruff before every commit.
Tests use a fake LLM and a real temporary Chroma store, so no Ollama is needed.
