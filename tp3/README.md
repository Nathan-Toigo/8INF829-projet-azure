# ThreeTokens Care Agent (tp3)

Streamlit multi-agent clinical care-plan assistant. A patient record is
uploaded, parsed by an LLM OCR step, stored, indexed, and analyzed by autonomous
AI agents that collaborate through shared memory to produce a structured care
plan.

> Synthetic data only - software validation, **not** for clinical use.

This is the **foundation build (Phases 1-4)**: scaffold + OpenRouter LLM/OCR/
embeddings + MongoDB + ChromaDB + LangGraph orchestrator and the first
autonomous agents (Timeline, Guidelines, Risk, Case Investigator). Reasoning,
care-plan, patient-adaptation, and learning agents (Phases 5-8) are scaffolded
as deferred stubs.

## Technology stack

| Layer | Choice |
|-------|--------|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM / OCR / embeddings gateway | OpenRouter (single key) |
| Vector DB | ChromaDB |
| Database | MongoDB (Docker) |
| Observability | LangSmith |
| Memory | Short-term state + cross-agent long-term (MongoDB + ChromaDB) |

A single `OPENROUTER_API_KEY` powers chat (tiered strong/small models), vision
OCR, and embeddings via OpenRouter's OpenAI-compatible endpoints.

## Architecture

```
Streamlit UI
  -> Upload -> LLM OCR (OpenRouter vision) -> MongoDB + ChromaDB indexing
  -> Clinical Question
       -> 1.1 Orchestrator (supervisor, routes + loop protection)
            -> 2.1 Timeline -> 2.2 Guidelines -> 2.3 Risk -> 2.4 Case Investigator
       (shared short-term memory; cross-agent long-term memory)
  -> Care Plan + Agent Trace (+ LangSmith)
```

- **Short-term memory** (`memory/`): the LangGraph state for one run; fully
  accessible and writable by every agent.
- **Long-term memory** (`memory/long_term_memory.py`): persistent, cross-run,
  readable by every agent; written only by the Guidelines and Case Investigator
  agents this pass.

## Setup

> Python 3.12 or 3.13 recommended (`chromadb` -> `onnxruntime` has no 3.14 wheel
> yet). `init.sh` picks a compatible interpreter when available.

```bash
cd tp3
./init.sh                       # create .venv, install deps, copy .env
# edit .env and set OPENROUTER_API_KEY (and optionally LangSmith)
docker compose up -d            # start MongoDB on localhost:27017
source .venv/bin/activate
streamlit run app/main.py
```

Optional: seed example clinical guidelines for the Guidelines Agent.

```bash
python -m ingestion.seed_guidelines
```

## Demo flow

1. **Upload** page -> "Seed clinical guidelines", then "Load sample patient from
   docs/" (or upload your own PDFs/images).
2. **Patient Profile** / **Timeline** -> review extracted context.
3. **Clinical Question** -> pick an example question and run the workflow.
4. **Care Plan** / **Agent Trace** -> review results, agent routing, tool calls,
   token usage, and the LangSmith link.

## Project layout

```
app/            Streamlit entry + 6 pages
agents/         base contract + orchestrator + foundation agents + stubs
graphs/         LangGraph main graph (orchestrator routing + loop protection)
memory/         short-term + cross-agent long-term memory
tools/          MongoDB + ChromaDB access
ingestion/      upload pipeline, LLM OCR, clinical extraction, chunking
core/           OpenRouter LLM wrapper (tiers + token ledger + ReAct loop)
config/         settings (env, models, LangSmith)
```

## Environment variables

See [.env.example](.env.example). Key ones: `OPENROUTER_API_KEY`,
`MONGODB_URI`, `CHROMA_DIR`, and the optional `LANGCHAIN_*` LangSmith settings.

## Notes

- ChromaDB collections: `patient_documents`, `clinical_guidelines`,
  `similar_cases`, `agent_learnings`.
- MongoDB collections: `patients`, `documents`, `clinical_resources`,
  `care_plans`, `agent_runs`, `audit_events`, `memory_snapshots`.
- Loop protection: `MAX_AGENT_STEPS=20`, `MAX_AGENT_RETRIES=2`.
