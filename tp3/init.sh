#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ThreeTokens care agent setup ==="

PYBIN=""
for cand in python3.12 python3.13 python3; do
  if command -v "$cand" &>/dev/null; then PYBIN="$cand"; break; fi
done

echo "Using interpreter: $PYBIN ($($PYBIN --version 2>&1))"

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  "$PYBIN" -m venv .venv
fi

# activation compatible Windows + Linux
if [[ -f .venv/Scripts/activate ]]; then
  source .venv/Scripts/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "Installing Python dependencies..."

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example"
  cp .env.example .env
fi

echo "=== Setup complete ==="