# Exam Q&A Generator: small, non-root, reproducible image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HOME=/home/app \
    DATA_DIR=/app/data \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run as an unprivileged user; /app/data is where the vector store and results database live.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app /home/app
WORKDIR /app

# Pinned dependencies (requirements.lock) so every build gets the same versions.
COPY requirements.lock ./
RUN pip install -r requirements.lock

USER app

# Download the default embedding model (~80 MB) now, so the first upload does not wait for it and the
# container works even if it is later started without internet access.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])" \
    && rm -f /home/app/.cache/chroma/onnx_models/*/onnx.tar.gz

COPY --chown=app:app app ./app
EXPOSE 8501

# No curl in the image: use Python for the health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
