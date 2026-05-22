import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import qdrant_client

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="rag_test_azure")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Loading documents...")
docs = SimpleDirectoryReader(str(DOCS_DIR)).load_data()
print(f"Loaded {len(docs)} documents")

print("Building index...")
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

print("Querying...")
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What are the main clinical findings in the documents?")
print("\n--- RESPONSE ---")
print(response)
print("\n--- SOURCES ---")
for node in response.source_nodes:
    print(f"- {node.metadata.get('file_name')}: score={node.score:.3f}")