import argparse
from pathlib import Path

from pipeline import build_pipeline, CONFIGS

DEFAULT_QUESTIONS_FILE = Path(__file__).parent.parent / "evaluation" / "team_questions.txt"


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Comparatif des methodes RAG sur des questions")
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS_FILE),
        help="Fichier texte, une question par ligne",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire l'index (apres ajout de documents)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS.keys()),
        help="Configs a comparer (par defaut: toutes)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "evaluation" / "comparatif.md"),
        help="Fichier de sortie markdown",
    )
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    print(f"{len(questions)} questions chargees.")

    # On construit chaque pipeline une seule fois et on reconstruit l'index
    # uniquement sur la premiere config.
    engines = {}
    for i, cfg_name in enumerate(args.configs):
        print(f"Preparation de la config: {cfg_name}")
        engines[cfg_name] = build_pipeline(CONFIGS[cfg_name], rebuild=(args.rebuild and i == 0))

    lines = ["# Comparatif des methodes RAG\n"]

    for qi, question in enumerate(questions, start=1):
        print(f"\n=== Q{qi}: {question} ===")
        lines.append(f"\n## Q{qi}. {question}\n")
        for cfg_name in args.configs:
            response = engines[cfg_name].query(question)
            answer_text = str(response).strip()
            srcs = ", ".join(
                f"{n.metadata.get('source')} p.{n.metadata.get('page', '-')}"
                for n in response.source_nodes
            )
            print(f"  [{cfg_name}] {answer_text[:80]}...")
            lines.append(f"**{cfg_name}**")
            lines.append(f"{answer_text}")
            lines.append(f"_Sources : {srcs}_\n")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparatif ecrit dans : {out_path}")


if __name__ == "__main__":
    main()