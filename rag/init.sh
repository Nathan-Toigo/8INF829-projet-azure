#!/usr/bin/env bash
# Setup Python venv, Ollama (Docker), and pull models for rag/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
CHAT_MODEL="${OLLAMA_CHAT_MODEL:-llama3.2}"

echo "=== RAG (Ollama + Docker) setup ==="

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

# shellcheck disable=SC1091
set -a
source .env
set +a
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-$EMBED_MODEL}"
CHAT_MODEL="${OLLAMA_CHAT_MODEL:-$CHAT_MODEL}"

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

if [[ -f .venv/Scripts/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/Scripts/activate
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Starting Ollama via Docker Compose..."
docker compose up -d

echo "Waiting for Ollama API..."
for _ in $(seq 1 30); do
  if curl -sf "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "Pulling models (first run may take several minutes)..."
docker compose exec -T ollama ollama pull "$EMBED_MODEL"
docker compose exec -T ollama ollama pull "$CHAT_MODEL"

echo ""
echo "=== Setup complete ==="
echo "  1. Optional: edit rag/.env (models, paths)"
echo "  2. Activate venv: source .venv/Scripts/activate  (or bin/activate)"
echo "  3. python run.py ingest"
echo "  4. python run.py ask \"Your question here\""
echo "     python run.py chat"
