import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Dataset
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_CHAT_DEPLOYMENT,
    EMBED_MODEL_NAME,
)

RESULTS_DIR = Path(__file__).parent / "results"
RAGAS_DIR = Path(__file__).parent / "ragas_results"
RAGAS_DIR.mkdir(parents=True, exist_ok=True)


def build_judge():
    judge_llm = AzureChatOpenAI(
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )
    judge_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


def load_results(cfg_name: str):
    path = RESULTS_DIR / f"{cfg_name}.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_config(cfg_name: str, llm_wrapper, emb_wrapper):
    rows = load_results(cfg_name)
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=emb_wrapper,
    )
    return result


def main():
    judge_llm, judge_emb = build_judge()

    configs = ["dense_top5", "bm25_top5", "hybrid_top5", "hybrid_rerank_top3", "dense_rerank_top3"]
    summary_rows = []

    for cfg_name in configs:
        print(f"\n=== Evaluating {cfg_name} ===")
        result = evaluate_config(cfg_name, judge_llm, judge_emb)
        df = result.to_pandas()
        out_path = RAGAS_DIR / f"{cfg_name}_ragas.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved per-question scores to {out_path.name}")

        scores = {}
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in df.columns:
                scores[col] = round(df[col].mean(), 3)
        scores["config"] = cfg_name
        summary_rows.append(scores)
        print(f"  Averages: {scores}")

    summary_df = pd.DataFrame(summary_rows).set_index("config")
    summary_path = RAGAS_DIR / "summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\n=== Final summary ===\n{summary_df}")
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()