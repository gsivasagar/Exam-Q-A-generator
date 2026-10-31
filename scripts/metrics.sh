#!/usr/bin/env bash
# Reproduce the numbers in docs/BASELINE_METRICS.md. Run from the repo root.
set -euo pipefail
echo "== size ==";        uvx --quiet radon raw -s app tests | grep -E "^(\*\* Total|[a-z].*\.py)|LOC|SLOC" | tail -8
echo "== complexity ==";  uvx --quiet radon cc -s -a app | tail -3
echo "== maintainability =="; uvx --quiet radon mi -s app
echo "== lint (E,F) ==";  uvx --quiet ruff check --isolated --select E,F --line-length 120 app tests --statistics || true
echo "== tests + coverage =="
uvx --quiet --python 3.13 --with-requirements requirements-dev.txt --with pytest-cov \
  pytest -q --cov=app --cov-branch -p no:cacheprovider
rm -f .coverage
echo "== chunking =="; python3 -I scripts/chunk_stats.py "${1:-tmp}" 2>/dev/null || .venv/bin/python -I scripts/chunk_stats.py "${1:-tmp}"
