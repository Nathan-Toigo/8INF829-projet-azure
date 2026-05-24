# Agent IA de synthèse et de personnalisation des recommandations cliniques

Phase 1 — Système RAG (Retrieval-Augmented Generation) appliqué à des documents cliniques.

Cours 8INF829 — Projet en trois phases : **RAG → Agent avec MCP → Orchestration agentique**.

## Objectif

Construire un système RAG capable d'ingérer des documents cliniques hétérogènes (rapports d'imagerie, comptes-rendus de consultation, résultats de laboratoire, dossier patient) et de répondre à des questions médicales en s'appuyant sur ces sources. Le projet compare plusieurs stratégies de recherche et de reranking, et évalue chacune avec des métriques objectives.

## Architecture

Le système suit une architecture **hybride** : composants de recherche en local, génération déléguée à Azure OpenAI.

| Composant | Technologie | Localisation |
|---|---|---|
| Parsing PDF | PyMuPDF | Local |
| Parsing DOCX | docx2txt / python-docx | Local |
| Chunking | LlamaIndex SentenceSplitter (512 / overlap 64) | Local |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) | Local |
| Vector store | Qdrant | Local (Docker) |
| Recherche | Dense / BM25 / Hybride (RRF) | Local |
| Reranking | BAAI/bge-reranker-base (cross-encoder) | Local |
| Génération (LLM) | Azure OpenAI gpt-4.1-mini | Azure |
| Évaluation | RAGAS | Local + juge Azure |

### Justification de l'architecture hybride

Le projet visait initialement un déploiement 100 % local avec Ollama. Les contraintes matérielles (16 Go de RAM, absence de GPU NVIDIA) limitaient la qualité du modèle de génération exploitable localement. Le choix s'est porté sur une architecture hybride : la recherche, l'indexation et le reranking restent locaux et gratuits, tandis que la génération est déléguée à Azure OpenAI pour obtenir une qualité de réponse adaptée au domaine clinique.

## Corpus

11 documents médicaux relatifs au suivi d'un patient chronique :

- 10 PDF cliniques : consultation ORL, échographie cervicale, pathologie FNA, IRM tête/cou, endoscopie GI, scanner thoracique, consultation pneumologie, tendances de laboratoire (2021–2025), bilan urologique, suivi médecine interne 2025
- 1 dossier patient (DOCX)

Après ingestion et chunking : **22 documents** → **41 chunks**.

## Structure du projet

```
8INF829-projet-azure/
├── docs/                       Corpus médical (PDF + DOCX)
├── src/
│   ├── config.py               Configuration centralisée
│   ├── ingestion.py            Parsing PDF/DOCX + métadonnées
│   ├── chunking.py             Découpage en chunks
│   ├── indexing.py             Construction de l'index Qdrant
│   ├── retrieval.py            Recherche dense, BM25, hybride RRF
│   ├── reranking.py            Reranking cross-encoder
│   └── pipeline.py             Orchestration et configurations
├── evaluation/
│   ├── questions.jsonl         20 questions de référence + ground truth
│   ├── run_queries.py          Génère les réponses pour chaque config
│   ├── run_ragas.py            Évaluation RAGAS des configs
│   ├── results/                Réponses brutes par configuration
│   └── ragas_results/          Scores RAGAS + summary.csv
├── .env                        Identifiants Azure (non versionné)
├── .gitignore
├── requirements.txt
└── README.md
```

## Prérequis

- Python 3.11
- Docker Desktop
- Un compte Azure OpenAI avec un déploiement de modèle de chat
- Windows 11 / PowerShell (testé)

## Installation

### 1. Cloner le dépôt

```
git clone <url-du-repo>
cd 8INF829-projet-azure
```

### 2. Créer et activer l'environnement virtuel

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```
pip install -r requirements.txt
pip install docx2txt
pip install llama-index-llms-azure-openai==0.2.2
pip install llama-index-retrievers-bm25==0.4.0
pip install ragas==0.2.6 langchain-openai==0.2.14 langchain-huggingface==0.1.2 datasets
```

### 4. Démarrer Qdrant

```
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Tableau de bord Qdrant : http://localhost:6333/dashboard

### 5. Configurer les variables d'environnement

Créer un fichier `.env` à la racine :

```
AZURE_OPENAI_API_KEY=votre_cle
AZURE_OPENAI_ENDPOINT=https://votre-ressource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_CHAT_DEPLOYMENT=gpt-4.1-mini
```

## Utilisation

À chaque session, vérifier que Qdrant tourne (`docker start qdrant`) et activer le venv.

### Construire l'index et poser des questions

```
cd src
python pipeline.py
```

Par défaut, utilise la configuration `hybrid_rerank_top3`. Pour changer, modifier la ligne `cfg = CONFIGS["..."]` en fin de fichier. Configurations disponibles : `dense_top5`, `bm25_top5`, `hybrid_top5`, `hybrid_rerank_top3`, `dense_rerank_top3`.

### Tester les composants individuellement (sans appel Azure)

```
python ingestion.py      # parsing
python chunking.py       # découpage
python retrieval.py      # comparaison des 3 modes de recherche
python reranking.py      # effet du reranker
```

### Lancer l'évaluation complète

```
cd ..\evaluation
python run_queries.py    # génère les réponses des 5 configs
python run_ragas.py      # évalue avec RAGAS
```

## Stratégies comparées

Cinq configurations ont été évaluées sur 20 questions de référence :

1. **dense_top5** — recherche vectorielle seule
2. **bm25_top5** — recherche lexicale seule
3. **hybrid_top5** — fusion dense + BM25 par Reciprocal Rank Fusion
4. **hybrid_rerank_top3** — hybride suivi d'un reranking cross-encoder
5. **dense_rerank_top3** — dense suivi d'un reranking cross-encoder

## Résultats (RAGAS)

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| dense_top5 | 1.000 | 0.632 | 0.733 | 0.900 |
| bm25_top5 | 0.985 | 0.643 | 0.692 | 0.842 |
| hybrid_top5 | 0.988 | 0.720 | 0.726 | 0.900 |
| **hybrid_rerank_top3** | **0.997** | **0.826** | **0.783** | 0.892 |
| dense_rerank_top3 | 1.000 | 0.739 | 0.642 | 0.842 |

### Lecture des résultats

- La recherche **hybride** dépasse la recherche dense seule sur la pertinence des réponses (0.720 contre 0.632), la fusion lexicale + vectorielle récupérant des passages que le dense seul manque.
- Le **reranking** apporte le gain le plus marqué : la configuration hybride passe de 0.720 à 0.826 en pertinence des réponses.
- La **faithfulness** reste proche de 1.0 pour toutes les configurations, ce qui est cohérent avec un corpus factuel et un modèle peu sujet aux hallucinations sur ce type de contenu.
- Le reranking ajoute environ 5 secondes de latence par requête sur CPU : un compromis qualité/latence à considérer selon le contexte d'usage.

## Limites connues

- Corpus de petite taille (11 documents), ce qui limite la portée statistique des comparaisons.
- Reranking exécuté sur CPU, donc lent. Un GPU réduirait significativement la latence.
- Les ground truth des questions de référence ont été construites à partir du contenu des documents et méritent une validation médicale plus poussée pour un usage réel.
- Le traitement des tableaux reste basique (extraction texte). Une extraction structurée dédiée (par exemple via Docling) constituerait une amélioration.

## Ancrage scientifique

Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv:2312.10997.

Ce travail de synthèse couvre l'ensemble des variantes comparées dans le projet (stratégies de découpage, recherche dense, recherche hybride, reranking) et sert de référence pour justifier les choix d'architecture.

## Prochaines phases

- **Phase 2** : transformation du RAG en agent doté d'outils via MCP (Model Context Protocol).
- **Phase 3** : orchestration agentique complète (système multi-agents).