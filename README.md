# Exam Q&A Generator with Ollama 📚

An interactive tool that generates educational quizzes from your study PDF notes using local LLMs via [Ollama](https://ollama.com).

## ✨ Features
*   **PDF Ingestion**: Upload your study materials directly.
*   **AI Question Generation**: Automatically creates quiz questions based on your content.
*   **Interactive Quiz**: Take the quiz directly in your browser.
*   **Auto-Grading**: Get instant feedback and scoring on your answers.
*   **Performance Tracking**: Track your progress over time.

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

> **Tip**: The database (`quiz_results.db`) and vector store are persisted in your local folder so you won't lose data when restarting the container.

## 🛠 Manual Installation

If you prefer running without Docker:

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    ```bash
    streamlit run app/ui.py
    ```

## 🔧 Configuration
*   **Model**: Defaults to `gemma3:latest`. Change `OLLAMA_MODEL` in `docker-compose.yml` or `.env` to use a different model.
*   **Ollama URL**: Defaults to `http://localhost:11434`.
