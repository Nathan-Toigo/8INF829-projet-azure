#!/usr/bin/env bash
# Verify Ollama container GPU access (run from nt-rag/ after compose up).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a
  source .env
  set +a
fi

# shellcheck source=scripts/detect-gpu.sh
source "$ROOT/scripts/detect-gpu.sh"

mapfile -t COMPOSE_ARGS < <(compose_file_args)
if [[ ${#COMPOSE_ARGS[@]} -eq 0 ]]; then
  exit 1
fi

echo "--- GPU verification ---"

if [[ "${COMPOSE_ARGS[*]}" == *"docker-compose.gpu.yml"* ]]; then
  echo "Compose: GPU override enabled"
  if docker compose "${COMPOSE_ARGS[@]}" exec -T ollama nvidia-smi 2>/dev/null; then
    echo "Container: nvidia-smi OK"
  else
    echo "Container: nvidia-smi not available (check Docker Desktop GPU support)"
  fi
else
  echo "Compose: CPU only (no GPU override)"
fi

echo ""
echo "Ollama loaded models (check PROCESSOR column after running a query):"
docker compose "${COMPOSE_ARGS[@]}" exec -T ollama ollama ps 2>/dev/null || \
  echo "  (no models loaded yet — run: python run.py ask \"test\")"

echo "--- end ---"
