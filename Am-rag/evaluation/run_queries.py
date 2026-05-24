import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import build_pipeline, CONFIGS

QUESTIONS_FILE = Path(__file__).parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_config(cfg_name: str, questions: list, rebuild: bool):
    cfg = CONFIGS[cfg_name]
    qe = build_pipeline(cfg, rebuild=rebuild)

    results = []
    total_latency = 0.0
    for q in questions:
        start = time.perf_counter()
        response = qe.query(q["question"])
        elapsed = time.perf_counter() - start
        total_latency += elapsed

        sources = [
            {
                "source": n.metadata.get("source"),
                "page": n.metadata.get("page"),
                "score": float(n.score) if n.score is not None else None,
            }
            for n in response.source_nodes
        ]
        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "expected_sources": q["expected_sources"],
            "answer": str(response),
            "retrieved_sources": sources,
            "contexts": [n.text for n in response.source_nodes],
            "latency_s": round(elapsed, 3),
        })
        print(f"  [{q['id']}] {elapsed:.2f}s")

    out = RESULTS_DIR / f"{cfg_name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    avg_latency = total_latency / len(questions)
    print(f"  -> {out.name}  avg latency: {avg_latency:.2f}s")
    return avg_latency


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    summary = {}
    for i, cfg_name in enumerate(CONFIGS.keys()):
        print(f"\n=== Running config: {cfg_name} ===")
        avg = run_config(cfg_name, questions, rebuild=(i == 0))
        summary[cfg_name] = avg

    print("\n=== Latency summary ===")
    for k, v in summary.items():
        print(f"  {k:25s} {v:.2f}s")


if __name__ == "__main__":
    main()