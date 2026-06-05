#!/usr/bin/env bash
# Initialize Python venv and install dependencies for the dms-2 clinical agent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== dms-2 clinical agent setup ==="

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
  echo "Creating .env from .env.example - set OPENAI_API_KEY before running."
  cp .env.example .env
fi

echo ""
echo "=== Setup complete ==="
echo "  1. Edit .env and set OPENAI_API_KEY (and optionally TAVILY_API_KEY / LangSmith)."
echo "  2. source .venv/bin/activate"
echo "  3. Terminal A: python -m mcp_server.server     # standalone MCP server"
echo "  4. Terminal B: streamlit run ui/app.py          # Streamlit UI"
echo ""
