# Contributing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt
pre-commit install          # optional: runs ruff before each commit
```

## Checks (CI runs the same)
```bash
ruff check app tests scripts      # lint
mypy                              # types (config in pyproject.toml)
pytest -q --cov                   # tests; fails under the coverage floor in pyproject.toml
```
Tests use a fake LLM and a real temporary Chroma store, so no Ollama or API key is needed.

## Working on a change
- One focused change per branch/PR. Describe *why*, not just what.
- Add or update tests with the change. Prefer testing behaviour through `generate_qa_pairs`, `grade_question`,
  or the Streamlit `AppTest` flows in `tests/test_ui.py`.
- New behaviour that could change quiz quality goes **behind a setting**, off (or old behaviour) by default, with
  the setting documented in `.env.example` and the README. Record measurements in `docs/BASELINE_METRICS.md`.
- Record non-obvious choices in `docs/DECISIONS.md`.
- Never put secrets in the repo. API keys come from the environment or the sidebar.

## Measuring
`scripts/metrics.sh` reproduces the code/test metrics. Other scripts in `scripts/` measure chunking, selection and
simulated latency; `docs/BASELINE_METRICS.md` explains how each number was produced.

## Docker
```bash
docker compose up --build                       # app, using the Ollama on your computer
docker compose --profile ollama up --build      # app + Ollama in Docker
```
The image uses the pinned `requirements.lock`. Regenerate it after changing `requirements.txt`:
`uv pip compile requirements.txt --python-version 3.11 --universal -o requirements.lock`
