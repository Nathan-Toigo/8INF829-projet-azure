#!/usr/bin/env bash
# Initialize Python venv, install dependencies, start pgvector Postgres, wait for DB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DMS environment setup ==="

# Python virtual environment
if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Environment file
if [[ ! -f .env ]]; then
  echo "Creating .env from .env.example — set OPENAI_API_KEY before running."
  cp .env.example .env
fi

# Docker Postgres + pgvector
if ! command -v docker &>/dev/null; then
  echo "ERROR: docker is not installed or not in PATH."
  exit 1
fi

echo "Starting PostgreSQL (pgvector) via Docker Compose..."
docker compose up -d

echo "Waiting for database to accept connections..."
python3 - <<'PY'
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from database import wait_for_db, init_schema

if not wait_for_db(max_attempts=40, delay_sec=2):
    print("Database did not become ready in time.")
    sys.exit(1)
init_schema()
print("Database ready; pgvector schema initialized.")
PY

echo ""
echo "=== Setup complete ==="
echo "  1. Edit .env and set OPENAI_API_KEY"
echo "  2. source .venv/bin/activate"
echo "  3. python run.py"
echo ""
