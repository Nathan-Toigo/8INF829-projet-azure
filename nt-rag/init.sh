#!/usr/bin/env bash
# Setup Python venv, Ollama (Docker), and pull models for nt-rag/
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

# shellcheck source=scripts/detect-gpu.sh
source "$ROOT/scripts/detect-gpu.sh"

mapfile -t COMPOSE_ARGS < <(compose_file_args)
rc=$?
if [[ $rc -eq 2 ]]; then
  exit 1
fi

if [[ "${COMPOSE_ARGS[*]}" == *"docker-compose.gpu.yml"* ]]; then
  echo "Ollama mode: GPU NVIDIA"
else
  echo "Ollama mode: CPU (no NVIDIA GPU usable by Docker)"
fi

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

OLLAMA_CONTAINER="${OLLAMA_CONTAINER_NAME:-rag_ollama}"

wait_for_ollama_api() {
  echo "Waiting for Ollama API..."
  for _ in $(seq 1 30); do
    if curl -sf "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "WARNING: Ollama API did not respond in time." >&2
  return 1
}

pull_ollama_models() {
  echo "Pulling models (first run may take several minutes)..."
  docker compose "${COMPOSE_ARGS[@]}" exec -T ollama ollama pull "$EMBED_MODEL"
  docker compose "${COMPOSE_ARGS[@]}" exec -T ollama ollama pull "$CHAT_MODEL"
}

echo ""
if docker inspect "$OLLAMA_CONTAINER" >/dev/null 2>&1; then
  if docker inspect -f '{{.State.Running}}' "$OLLAMA_CONTAINER" | grep -qx true; then
    echo "Ollama container '$OLLAMA_CONTAINER' is already running — skipping Docker startup."
  else
    echo "Ollama container '$OLLAMA_CONTAINER' exists (stopped) — starting it."
    docker start "$OLLAMA_CONTAINER"
    wait_for_ollama_api || true
  fi
else
  echo "Starting Ollama via Docker Compose..."
  docker compose "${COMPOSE_ARGS[@]}" up -d
  wait_for_ollama_api || true
  pull_ollama_models
  bash "$ROOT/scripts/verify-ollama-gpu.sh" || true
fi

echo ""
echo "=== Setup complete ==="
echo "  1. Optional: edit nt-rag/.env (models, OLLAMA_GPU_MODE, paths)"
echo "  2. Activate venv: source .venv/Scripts/activate  (or bin/activate)"
echo "  3. python run.py ingest"
echo "  4. python run.py ask \"Your question here\""
echo "     python run.py chat"
