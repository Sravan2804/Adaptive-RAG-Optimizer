import json
import os
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Optimize HuggingFace Hub environment
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = Path(__file__).resolve().parent.parent

def get_embedding_function():
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        from sklearn.feature_extraction.text import HashingVectorizer
        class LocalFallbackEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, n_features: int = 384):
                self.vectorizer = HashingVectorizer(n_features=n_features)
            def __call__(self, input):
                return self.vectorizer.transform(input).toarray().tolist()
        return LocalFallbackEmbeddingFunction()

def build_vector_store():
    kb_path = Path("data/knowledge_base.json") if Path("data/knowledge_base.json").exists() else BASE_DIR / "data" / "knowledge_base.json"
    db_path = Path("./chroma_db") if Path("data/knowledge_base.json").exists() else BASE_DIR / "chroma_db"
    
    if not kb_path.exists():
        raise FileNotFoundError(f"Missing {kb_path} corpus file.")
        
    client = chromadb.PersistentClient(path=str(db_path))
    ef = get_embedding_function()
    
    collection = client.get_or_create_collection(
        name="company_docs",
        embedding_function=ef
    )
    
    with open(kb_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
        
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs]
    )
    print(f"[SUCCESS] Ingested {len(docs)} documents into ChromaDB at {db_path}.")

if __name__ == "__main__":
    build_vector_store()
