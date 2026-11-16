import json
import os
import chromadb
from chromadb.utils import embedding_functions

def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Support running either from root or from src
    if os.path.exists("data/knowledge_base.json"):
        kb_path = "data/knowledge_base.json"
        db_path = "./chroma_db"
    else:
        kb_path = os.path.join(base_dir, "data", "knowledge_base.json")
        db_path = os.path.join(base_dir, "chroma_db")
    return kb_path, db_path

def build_vector_store():
    kb_path, db_path = get_paths()
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name="company_docs",
        embedding_function=ef
    )
    
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Missing {kb_path} corpus file.")
        
    with open(kb_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
        
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs]
    )
    print(f"[SUCCESS] Ingested {len(docs)} documents into ChromaDB at {db_path}.")

if __name__ == "__main__":
    build_vector_store()
