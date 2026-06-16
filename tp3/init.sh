#!/usr/bin/env bash
# Initialize Python venv and install dependencies for the ThreeTokens care agent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ThreeTokens care agent setup ==="

# chromadb depends on onnxruntime, which has no wheels for Python 3.14 yet.
# Prefer a 3.12/3.13 interpreter when available.
PYBIN=""
for cand in python3.12 python3.13 python3; do
  if command -v "$cand" &>/dev/null; then PYBIN="$cand"; break; fi
done
echo "Using interpreter: $PYBIN ($($PYBIN --version 2>&1))"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example - set OPENROUTER_API_KEY before running."
  cp .env.example .env
fi

echo ""
echo "=== Setup complete ==="
echo "  1. Edit .env and set OPENROUTER_API_KEY (and optionally LangSmith)."
echo "  2. docker compose up -d            # start MongoDB"
echo "  3. source .venv/bin/activate"
echo "  4. streamlit run app/main.py       # launch the Streamlit app"
echo ""
