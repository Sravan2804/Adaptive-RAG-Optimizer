import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env before setting defaults or importing Hugging Face / ChromaDB modules
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)
else:
    load_dotenv(override=False)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import chromadb
from chromadb.utils import embedding_functions

BACKEND_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND_FALLBACK = "fallback-hashing-384"

class LocalFallbackEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, n_features: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vectorizer = HashingVectorizer(n_features=n_features)

    def __call__(self, input):
        return self.vectorizer.transform(input).toarray().tolist()

def get_embedding_function(force_fallback: bool = False) -> Tuple[embedding_functions.EmbeddingFunction, str]:
    if not force_fallback:
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            return ef, BACKEND_MINILM
        except Exception:
            pass
    return LocalFallbackEmbeddingFunction(), BACKEND_FALLBACK

def resolve_db_path(db_path: Optional[str] = None, corpus_root: Optional[Path] = None) -> Path:
    if db_path:
        return Path(db_path)
    root = corpus_root if corpus_root is not None else (Path(".") if Path("data/knowledge_base.json").exists() else BASE_DIR)
    cwd_db = Path("chroma_db")
    base_db = root / "chroma_db"
    if cwd_db.exists() and (cwd_db / "chroma.sqlite3").exists():
        return cwd_db
    if base_db.exists() and (base_db / "chroma.sqlite3").exists():
        return base_db
    return root / "chroma_db"

def compute_corpus_revision(docs) -> str:
    content = json.dumps(sorted(docs, key=lambda d: str(d.get("id", ""))), sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

def build_vector_store(db_path: Optional[str] = None, kb_path: Optional[str] = None):
    resolved_kb = Path(kb_path) if kb_path else (Path("data/knowledge_base.json") if Path("data/knowledge_base.json").exists() else BASE_DIR / "data" / "knowledge_base.json")
    corpus_root = resolved_kb.parent.parent
    resolved_db = resolve_db_path(db_path, corpus_root=corpus_root)
    
    if not resolved_kb.exists():
        raise FileNotFoundError(f"Missing {resolved_kb} corpus file.")
        
    with open(resolved_kb, "r", encoding="utf-8") as f:
        docs = json.load(f)

    ids = [d["id"] for d in docs]
    documents = [d["text"] for d in docs]
    corpus_rev = compute_corpus_revision(docs)
        
    client = chromadb.PersistentClient(path=str(resolved_db))
    ef, backend_name = get_embedding_function()
    
    # Handle Chroma 0.6.x (strings) and <=0.5.x (objects) list_collections()
    raw_cols = client.list_collections()
    existing_col_names = {c if isinstance(c, str) else c.name for c in raw_cols}
    
    target_collection_name = "company_docs"

    if "company_docs" in existing_col_names:
        existing_col = client.get_collection("company_docs")
        stored_backend = (existing_col.metadata or {}).get("embedding_backend")
        if stored_backend is None or stored_backend != backend_name:
            # Backend is unknown or differs: preserve legacy collection and use separate backend-tagged index
            target_collection_name = "company_docs_fallback" if backend_name == BACKEND_FALLBACK else f"company_docs_{backend_name.replace('/', '_').replace('-', '_')}"

    collection = client.get_or_create_collection(
        name=target_collection_name,
        embedding_function=ef,
        metadata={"embedding_backend": backend_name, "status": "building"}
    )
    collection.upsert(ids=ids, documents=documents)
    # B.4: Remove stale documents (IDs present in DB but not in incoming corpus)
    existing_ids = collection.get(include=[])["ids"]
    stale_ids = list(set(existing_ids) - set(ids))
    if stale_ids:
        collection.delete(ids=stale_ids)
        print(f"[INFO] Removed {len(stale_ids)} stale document(s) from '{target_collection_name}'.")
    # Verify count matches before marking ready
    actual_count = collection.count()
    if actual_count != len(ids):
        raise RuntimeError(
            f"Ingestion verification failed for '{target_collection_name}': "
            f"expected {len(ids)} documents, found {actual_count}. Leaving status='building' for retry."
        )
    collection.modify(metadata={
        "embedding_backend": backend_name,
        "corpus_revision": corpus_rev,
        "status": "ready"
    })
    print(f"[SUCCESS] Ingested {len(docs)} documents into collection '{target_collection_name}' ({backend_name}, rev:{corpus_rev}) at {resolved_db}.")

    # If primary backend succeeded, also build/upsert the fallback collection so offline/un-cached environments have a ready fallback index
    if backend_name != BACKEND_FALLBACK:
        try:
            fallback_ef, fallback_backend = get_embedding_function(force_fallback=True)
            fallback_col = client.get_or_create_collection(
                name="company_docs_fallback",
                embedding_function=fallback_ef,
                metadata={"embedding_backend": fallback_backend, "status": "building"}
            )
            # B.3: Validate backend tag before upserting into fallback collection
            fb_stored_backend = (fallback_col.metadata or {}).get("embedding_backend")
            if fb_stored_backend is not None and fb_stored_backend != fallback_backend:
                print(f"[WARNING] Fallback collection 'company_docs_fallback' has incompatible backend '{fb_stored_backend}'. Skipping fallback upsert.")
            else:
                # Upsert documents BEFORE marking active/ready so it remains inactive until completed
                fallback_col.upsert(ids=ids, documents=documents)
                stale_fallback_ids = list(set(fallback_col.get(include=[])["ids"]) - set(ids))
                if stale_fallback_ids:
                    fallback_col.delete(ids=stale_fallback_ids)
                fallback_col.modify(metadata={
                    "embedding_backend": fallback_backend,
                    "corpus_revision": corpus_rev,
                    "status": "ready"
                })
                print(f"[SUCCESS] Ingested {len(docs)} documents into fallback collection 'company_docs_fallback' ({fallback_backend}, rev:{corpus_rev}).")
        except Exception as exc:
            print(f"[WARNING] Fallback collection upsert failed (non-fatal): {exc}")

if __name__ == "__main__":
    build_vector_store()
