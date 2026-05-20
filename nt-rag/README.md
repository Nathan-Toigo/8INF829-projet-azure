# RAG — Ollama local over `docs/`

Build a retrieval-augmented index from PDF/DOCX files in the repository `docs/` folder, then ask questions with **Ollama** running locally in **Docker** (embeddings + chat). No Azure account required.

## Prerequisites

- Python 3.11+
- [Docker](https://docs.docker.com/get-docker/) (Desktop or Engine)
- ~4–8 GB disk for models (depends on chat model choice)

## Quick start

```bash
cd rag
./init.sh
source .venv/Scripts/activate   # Git Bash on Windows
# or: source .venv/bin/activate  # Linux / macOS
python run.py ingest
python run.py ask "What follow-up was recommended on the 2022 CT chest?"
python run.py chat
```

`init.sh` starts Ollama (`docker compose up -d`) and pulls the default models.

## Ollama (Docker)

| Command | Action |
|---------|--------|
| `docker compose up -d` | Start Ollama on port 11434 |
| `docker compose down` | Stop container |
| `docker compose exec ollama ollama list` | Installed models |
| `docker compose exec ollama ollama pull <model>` | Add another model |

Default models (configurable in `.env`):

- **Embeddings:** `nomic-embed-text` (for vector search)
- **Chat:** `llama3.2` (answers from retrieved context)

Smaller chat alternative: set `OLLAMA_CHAT_MODEL=mistral` in `.env` and run `docker compose exec ollama ollama pull mistral`.

## Configuration (`rag/.env`)

| Variable | Description |
|----------|-------------|
| `OLLAMA_BASE_URL` | Default `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | Model for ingest / query embeddings |
| `OLLAMA_CHAT_MODEL` | Model for `ask` / `chat` |
| `DOCS_DIR` | Default `../docs` |

After changing the embedding model, re-run `python run.py ingest` (vector dimensions may differ).

## Commands

| Command | Action |
|---------|--------|
| `python run.py ingest` | Load docs, chunk, embed, store in Chroma |
| `python run.py ask "..."` | Single question |
| `python run.py chat` | Interactive session (`quit` to exit) |

Vector data: `rag/data/chroma/` (gitignored).

## Project layout

```
rag/
  docker-compose.yml  # Ollama container
  config.py
  ollama_client.py    # HTTP client for /api/embed and /api/chat
  documents.py
  chunking.py
  store.py
  ingest.py
  query.py
  run.py
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Cannot reach Ollama | `docker compose up -d` in `rag/` |
| Missing model | `docker compose exec ollama ollama pull <model>` |
| Slow ingest | Normal on CPU; reduce docs or use a smaller embed model |
| Empty / bad answers | Re-ingest; try a larger chat model |
