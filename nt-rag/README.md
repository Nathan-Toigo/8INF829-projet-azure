# RAG — Ollama local over `docs/`

Build a retrieval-augmented index from PDF/DOCX files in the repository `docs/` folder, then ask questions with **Ollama** running locally in **Docker** (embeddings + chat). No Azure account required.

## Prerequisites

- Python 3.11+
- [Docker Desktop](https://docs.docker.com/get-docker/) (WSL2 backend on Windows)
- ~4–8 GB disk for models (depends on chat model choice)
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

Vector data: `nt-rag/data/chroma/` (gitignored).

## Benchmark / evaluation pipeline

Compare **chunking methods**, **chat models**, and **host profiles** with timing and LLM judge scores.

### Quick run

```bash
pip install -r requirements.txt
python run.py eval --config experiments/benchmark.yaml --dry-run
python run.py eval --config experiments/benchmark.yaml
python run.py eval --only fixed_chars_llama32
```

### What is measured

| Category | Metrics |
|----------|---------|
| **Latency** | `ingest_total_sec`, `query_embed_ms`, `retrieve_ms`, `chat_ms`, `full_chart_sec`, `judge_sec` |
| **Quality (judge)** | `accuracy_score`, `quality_score`, `thoroughness_score`, `global_accuracy_score` vs full-chart answer |
| **Golden (optional)** | `golden_match_score` when question id exists in `eval/golden_answers.json` |

### Configuration

- [`experiments/benchmark.yaml`](experiments/benchmark.yaml) - experiment matrix (`chunk_method`, models, `top_k`)
- [`eval/questions.json`](eval/questions.json) - questions (`default_subset` for faster runs)
- [`eval/golden_answers.json`](eval/golden_answers.json) - reference answers for selected ids

Chunk methods: `fixed_chars`, `paragraph`, `page`, `words_250`.

Each experiment uses its own Chroma collection: `docs_rag_{chunk_method}_{embed_model}`.

### Outputs

Results under `nt-rag/results/<timestamp>/`:

- `hardware.json` - CPU, RAM, GPU snapshot
- `details.jsonl` - one JSON line per (experiment, question)
- `summary.csv` - averages per experiment
- `report.md` - readable table

Set `run_full_chart: false` or `run_judge: false` in YAML for faster smoke tests.

Compare runs on different PCs by setting `host_profile` in the YAML.

## Project layout

```
nt-rag/
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
