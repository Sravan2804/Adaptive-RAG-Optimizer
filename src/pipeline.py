import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
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

import joblib
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

BACKEND_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BACKEND_FALLBACK = "fallback-hashing-384"

ACTION_MAP = {
    0: {"name": "SKIP",   "k": 0, "tier_cost_usd": 0.0005},
    1: {"name": "NARROW", "k": 2, "tier_cost_usd": 0.0020},
    2: {"name": "DEEP",   "k": 6, "tier_cost_usd": 0.0060},
}

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

class FallbackTokenizer:
    def encode(self, text: str):
        # Reliable fallback: ~4 characters per token without allocating a list
        return range(max(1, len(text) // 4))

class AdaptiveRAGEngine:
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        model_path: Optional[str] = None,
        db_path: Optional[str] = None
    ):
        target_model = Path(model_path) if model_path else (Path("models/query_router.joblib") if Path("models/query_router.joblib").exists() else BASE_DIR / "models" / "query_router.joblib")
        self.router = joblib.load(target_model)
        self.confidence_threshold = confidence_threshold
        
        target_db = resolve_db_path(db_path)
        self.client = chromadb.PersistentClient(path=str(target_db))
        
        # Handle Chroma 0.6.x (strings) and <=0.5.x (objects) list_collections() safely
        raw_cols = self.client.list_collections()
        col_metadata = {}
        dummy_ef = LocalFallbackEmbeddingFunction()
        for c in raw_cols:
            c_name = c if isinstance(c, str) else c.name
            c_meta = c.metadata if not isinstance(c, str) else None
            if c_meta is None:
                try:
                    c_meta = self.client.get_collection(c_name, embedding_function=dummy_ef).metadata
                except Exception:
                    c_meta = {}
            col_metadata[c_name] = c_meta or {}

        minilm_col_name = f"company_docs_{BACKEND_MINILM.replace('/', '_').replace('-', '_')}"

        # Discover candidates
        # C.1: Validate MiniLM tag when found under the named collection
        minilm_candidate_name = None
        if minilm_col_name in col_metadata:
            if col_metadata[minilm_col_name].get("embedding_backend") == BACKEND_MINILM:
                minilm_candidate_name = minilm_col_name
            # else: named collection has wrong backend tag — skip it
        if minilm_candidate_name is None and "company_docs" in col_metadata:
            if col_metadata["company_docs"].get("embedding_backend") == BACKEND_MINILM:
                minilm_candidate_name = "company_docs"

        # C.2: Only accept company_docs_fallback as hashing candidate when status=="ready"
        hashing_conflict_reason = None
        hashing_candidate_name = None
        if "company_docs_fallback" in col_metadata:
            fb_meta = col_metadata["company_docs_fallback"]
            fb_backend = fb_meta.get("embedding_backend")
            fb_status = fb_meta.get("status", "ready")
            if fb_backend is not None and fb_backend != BACKEND_FALLBACK:
                # C.3: Incompatible backend — skip instead of raise, record reason
                hashing_conflict_reason = (
                    f"Collection 'company_docs_fallback' has incompatible backend '{fb_backend}', "
                    f"expected '{BACKEND_FALLBACK}'. Run ingest.py to rebuild."
                )
            elif fb_status == "ready":
                hashing_candidate_name = "company_docs_fallback"
            # else status=="building" — skip; check company_docs below
        if hashing_candidate_name is None and "company_docs" in col_metadata:
            if col_metadata["company_docs"].get("embedding_backend") == BACKEND_FALLBACK:
                hashing_candidate_name = "company_docs"

        legacy_candidate_name = None
        if "company_docs" in col_metadata:
            if col_metadata["company_docs"].get("embedding_backend") is None:
                legacy_candidate_name = "company_docs"

        # Track completed corpus revision
        minilm_rev = col_metadata.get(minilm_candidate_name, {}).get("corpus_revision") if minilm_candidate_name else None
        hashing_rev = col_metadata.get(hashing_candidate_name, {}).get("corpus_revision") if hashing_candidate_name else None

        minilm_is_ready = col_metadata.get(minilm_candidate_name, {}).get("status", "ready") == "ready" if minilm_candidate_name else False
        hashing_is_ready = hashing_candidate_name is not None  # already filtered by status above

        selected_col_name = None
        selected_ef = None

        # C.4: Prefer MiniLM if available, ready, and its revision matches hashing_rev (or no hashing index)
        if minilm_candidate_name and minilm_is_ready and (hashing_rev is None or minilm_rev == hashing_rev):
            try:
                ef, b_id = get_embedding_function(force_fallback=False)
                if b_id == BACKEND_MINILM:
                    selected_col_name = minilm_candidate_name
                    selected_ef = ef
            except Exception:
                selected_col_name = None

        # Fallback to hashing index if MiniLM cannot load
        if selected_col_name is None and hashing_candidate_name and hashing_is_ready:
            ef, _ = get_embedding_function(force_fallback=True)
            selected_col_name = hashing_candidate_name
            selected_ef = ef

        # C.5: Legacy untagged collection: require MiniLM; do NOT fall back to hashing for untagged index
        if selected_col_name is None and legacy_candidate_name:
            try:
                ef, b_id = get_embedding_function(force_fallback=False)
                if b_id == BACKEND_MINILM:
                    selected_col_name = legacy_candidate_name
                    selected_ef = ef
                else:
                    legacy_candidate_name = None  # hashing backend for untagged collection is unsafe
            except Exception:
                legacy_candidate_name = None

        if selected_col_name is None:
            detail = hashing_conflict_reason or "Run ingest.py to create the vector store."
            raise FileNotFoundError(
                f"No usable document collections found in ChromaDB at {target_db}. {detail}"
            )

        self.ef = selected_ef
        self.collection_name = selected_col_name
        self.collection = self.client.get_collection(self.collection_name, embedding_function=self.ef)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and not api_key.startswith("your_") and not api_key.startswith("mock"):
            self.llm = OpenAI(api_key=api_key)
        else:
            self.llm = None
            
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = FallbackTokenizer()

    def route_query(self, query: str) -> Dict[str, Any]:
        start = time.perf_counter()
        probs = self.router.predict_proba([query])[0]
        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])
        
        # Uncertainty guardrail fallback to NARROW retrieval
        if confidence < self.confidence_threshold:
            pred_class = 1
            
        router_ms = (time.perf_counter() - start) * 1000
        decision = ACTION_MAP[pred_class]
        return {
            "class_id": pred_class,
            "tier": decision["name"],
            "k": decision["k"],
            "tier_cost_usd": decision["tier_cost_usd"],
            "confidence": round(confidence, 3),
            "router_latency_ms": round(router_ms, 2)
        }

    def run(self, query: str, mode: str = "adaptive", static_k: int = 5) -> Dict[str, Any]:
        total_start = time.perf_counter()
        
        if mode == "adaptive":
            route_info = self.route_query(query)
            k = route_info["k"]
        else:
            route_info = {"tier": "STATIC", "k": static_k, "router_latency_ms": 0.0, "tier_cost_usd": 0.005}
            k = static_k

        retrieval_start = time.perf_counter()
        if k > 0:
            results = self.collection.query(query_texts=[query], n_results=k)
            chunks = results["documents"][0]
        else:
            chunks = []
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        context_block = "\n".join([f"- {c}" for c in chunks]) if chunks else "No external context provided."
        prompt = f"Answer the question accurately using the context if provided.\n\nContext:\n{context_block}\n\nQuestion: {query}"
        
        try:
            input_tokens = len(self.tokenizer.encode(prompt))
        except Exception:
            input_tokens = len(prompt.split()) * 4 // 3

        if self.llm is not None:
            try:
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"[LLM Error: {e}] Context: {chunks[:1]}"
        else:
            answer = f"[Direct/Contextual Synthesized Output] Answer for query '{query}' (context chunks: {len(chunks)})"

        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "mode": mode,
            "query": query,
            "answer": answer,
            "k_used": k,
            "input_tokens": input_tokens,
            "router_ms": route_info["router_latency_ms"],
            "retrieval_ms": round(retrieval_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "tier_cost_usd": route_info["tier_cost_usd"]
        }
