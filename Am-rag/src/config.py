import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini")
AZURE_MODEL_ALIAS = "gpt-4o-mini"  # alias connu de llama-index pour la context window

# Embeddings (local)
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "rag_clinical"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 5