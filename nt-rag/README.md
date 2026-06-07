# RAG ? Ollama local over `docs/`

Build a retrieval-augmented index from PDF/DOCX files in the repository `docs/` folder, then ask questions with **Ollama** running locally in **Docker** (embeddings + chat). No Azure account required.

## Prerequisites

- Python 3.11+
- [Docker Desktop](https://docs.docker.com/get-docker/) (WSL2 backend on Windows)
- ~4?8 GB disk for models (depends on chat model choice)
- **Optional (GPU):** NVIDIA GPU, recent drivers, GPU support enabled in Docker Desktop

## Quick start

```bash
cd nt-rag
./init.sh
source .venv/Scripts/activate   # Git Bash on Windows
# or: source .venv/bin/activate  # Linux / macOS
python run.py ingest
python run.py ask "What follow-up was recommended on the 2022 CT chest?"
python run.py chat
```

`init.sh` detects an NVIDIA GPU, starts Ollama with or without GPU, and pulls default models.

## GPU (Windows + Docker Desktop)

When an NVIDIA GPU is available to Docker, `init.sh` adds [`docker-compose.gpu.yml`](docker-compose.gpu.yml) (`gpus: all`).

### Prerequisites

1. NVIDIA drivers installed on Windows (`nvidia-smi` works in PowerShell or Git Bash).
2. Docker Desktop ? **Settings** ? General: use **WSL2** engine.
3. Docker Desktop ? **Settings** ? **Resources** or **Features**: enable **GPU** / **NVIDIA GPU support**.
4. If `docker run --gpus all` fails: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) in WSL2.

### Configuration (`.env`)

| Variable | Values | Description |
|----------|--------|-------------|
| `OLLAMA_GPU_MODE` | `auto` (default) | Use GPU if NVIDIA + Docker detect it |
| | `gpu` | Force GPU compose (error if unavailable) |
| | `cpu` | Force CPU even if GPU present |

### Manual start with GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

CPU only:

```bash
docker compose up -d
```

### Verify GPU

```bash
nvidia-smi
bash scripts/verify-ollama-gpu.sh
docker compose exec ollama ollama ps   # after one ask: look for GPU in PROCESSOR
```

### GPU troubleshooting

| Symptom | Fix |
|---------|-----|
| Always CPU mode | Set `OLLAMA_GPU_MODE=gpu` to see error; fix Docker GPU / drivers |
| `nvidia-smi` OK on host, not in container | Enable GPU in Docker Desktop; restart Docker |
| Ollama still 100% CPU in `ollama ps` | Run a query first; confirm GPU override was used |
| Out of VRAM | Use `OLLAMA_CHAT_MODEL=llama3.2:3b` and `ollama pull llama3.2:3b` |

## Ollama (Docker)

| Command | Action |
|---------|--------|
| `./init.sh` | Venv + detect GPU + start Ollama + pull models |
| `docker compose up -d` | Start Ollama (CPU) |
| `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` | Start with GPU |
| `docker compose down` | Stop container |
| `docker compose exec ollama ollama list` | Installed models |

Default models (configurable in `.env`):

- **Embeddings:** `nomic-embed-text`
- **Chat:** `llama3.2`

Smaller chat model (less VRAM): `OLLAMA_CHAT_MODEL=llama3.2:3b`

## Configuration (`nt-rag/.env`)

| Variable | Description |
|----------|-------------|
| `OLLAMA_BASE_URL` | Default `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | Model for ingest / query embeddings |
| `OLLAMA_CHAT_MODEL` | Model for `ask` / `chat` |
| `OLLAMA_GPU_MODE` | `auto`, `gpu`, or `cpu` |
| `DOCS_DIR` | Default `../docs` |

After changing the embedding model, re-run `python run.py ingest`.

## Commands

| Command | Action |
|---------|--------|
| `python run.py ingest` | Load docs, chunk, embed, store in Chroma |
| `python run.py ask "..."` | Single question |
| `python run.py chat` | Interactive session (`quit` to exit) |
| `python run.py eval` | Run benchmark matrix (see below) |

## Agent conversationnel (web + MCP)

Interface web et serveur MCP qui **r?utilisent** les pipelines existants (`ingest`, `query`, Chroma, Ollama). Les commandes CLI ci-dessus restent inchang?es.

### Demarrage

**Option A ? Docker Compose (Ollama + MCP + UI)** :

```bash
cd nt-rag
docker compose up -d --build
# UI : http://localhost:8501
# MCP : http://localhost:8010/mcp
# Ollama : http://localhost:11434
```

Premier lancement : tirer les modeles et indexer :

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
docker compose exec mcp-server python -c "from ingest import run_ingest; run_ingest(clear=True)"
```

**Option B ? terminaux locaux** (Ollama via Docker, agent en local) :

Terminal 1 ? serveur MCP (outils RAG + exemples) :

```bash
cd nt-rag
source .venv/Scripts/activate
python -m mcp_server.server
```

Terminal 2 ? interface web :

```bash
streamlit run ui/app.py
```

Ouvrir l'URL affich?e (souvent `http://localhost:8501`).

### Outils MCP expos?s

| Outil | R?le |
|-------|------|
| `rag_ask` | Pipeline RAG complet (`query.ask_with_metrics`) |
| `rag_search` | Recherche s?mantique sans g?n?ration |
| `ingest_documents` | Indexation `docs/` (`ingest.run_ingest`) |
| `index_stats` | Statistiques Chroma |
| `get_current_time` | Exemple : heure UTC |
| `count_words` | Exemple : comptage de mots |
| `echo_message` | Exemple : echo |
| `list_document_sources` | Liste les fichiers dans `docs/` |

Configuration MCP dans `.env` : `MCP_HOST`, `MCP_PORT` (d?faut `8010`).

### Layout agent

```
nt-rag/
  agent/           # Agent Ollama + boucle d'outils MCP
  mcp_server/      # Serveur HTTP MCP (FastMCP)
  ui/app.py        # Interface Streamlit
```

Vector data: `nt-rag/data/chroma/` (gitignored).

## Benchmark / evaluation pipeline

Compare **chunking methods** and **chat models** ? RAG answers only (no full-chart, no LLM judge).

### Quick run

```bash
pip install -r requirements.txt
python run.py eval --config experiments/benchmark.yaml --dry-run
python run.py eval --config experiments/benchmark.yaml
python run.py eval --only fixed_chars_llama32
```

By default `reuse_existing_collection: true` in the benchmark YAML skips re-embedding when the Chroma collection for that experiment already has vectors (e.g. after a previous `ingest` or eval). Force a full refresh:

```bash
python run.py eval --config experiments/benchmark.yaml --force-ingest
```

### What is measured (`summary.csv`)

| Column | Meaning |
|--------|---------|
| `chunk_method` | Chunking strategy |
| `question` | Question text |
| `chat_ms` | RAG answer generation time (ms) |
| `rag_answer` | Model answer |
| `total_tokens` | Ollama tokens (prompt + completion) |
| `top1_distance` | Cosine distance of best retrieved chunk |

### Configuration

- [`experiments/benchmark.yaml`](experiments/benchmark.yaml) - experiment matrix (`chunk_method`, models, `top_k`)
- [`eval/questions.json`](eval/questions.json) - questions (`default_subset` for faster runs)

Chunk methods: `fixed_chars`, `paragraph`, `page`, `words_250`.

Each experiment uses its own Chroma collection: `docs_rag_{chunk_method}_{embed_model}`. Re-ingest only when docs/chunking change or use `--force-ingest`.

### Outputs

Results under `nt-rag/results/<timestamp>/`:

- `summary.csv` - one row per (experiment, question): chunk method, question, chat ms, answer, tokens, top1 distance
- `report.md` - readable preview table
- `hardware.json` - CPU, RAM snapshot
- `config_resolved.yaml` - resolved benchmark config

Compare runs on different PCs by setting `host_profile` in the YAML.

## Project layout

```
nt-rag/
  agent/
    chat_agent.py
    mcp_client.py
    runtime.py
  mcp_server/
    server.py
    tools/
  ui/
    app.py
  docker-compose.yml
  docker-compose.gpu.yml
  experiments/benchmark.yaml
  eval/
    benchmark.py
    hardware.py
    judge.py
    questions.json
    golden_answers.json
  scripts/
  chunking.py
  ingest.py
  query.py
  run.py
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Cannot reach Ollama | `docker compose up -d` in `nt-rag/` |
| Missing model | `docker compose exec ollama ollama pull <model>` |
| Slow ingest / chat | Enable GPU; or use `llama3.2:3b` |
| Empty / bad answers | Re-ingest; try a larger chat model |
| `httpx.ReadError` / WinError 10054 during eval | Ollama closed the socket (often OOM or prompt too large). Lower `MAX_FULL_DOC_CHARS` / `max_full_doc_chars`; check `docker logs rag_ollama`; retries are automatic (`OLLAMA_CHAT_RETRIES`) |
| `400 Bad Request` on `/api/embed` during ingest | Chunk text too long for `nomic-embed-text` (common with `chunk_method: page`). Fixed by `EMBED_MAX_CHARS` truncation + page split in `chunking.py`; or disable the `page` experiment in `benchmark.yaml` |
| Judge `JSON parse failed` / empty accuracy in CSV | Judge output was truncated; parser now extracts numeric scores from partial JSON and prompts request scores-only |
