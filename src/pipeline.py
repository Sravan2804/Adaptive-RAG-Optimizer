import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import joblib
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent

ACTION_MAP = {
    0: {"name": "SKIP",   "k": 0, "tier_cost_usd": 0.0005},
    1: {"name": "NARROW", "k": 2, "tier_cost_usd": 0.0020},
    2: {"name": "DEEP",   "k": 6, "tier_cost_usd": 0.0060},
}

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

class FallbackTokenizer:
    def encode(self, text: str):
        # Reliable fallback: ~4 characters per token
        return list(range(max(1, len(text) // 4)))

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
        
        target_db = Path(db_path) if db_path else (Path("chroma_db") if Path("chroma_db").exists() else BASE_DIR / "chroma_db")
        self.client = chromadb.PersistentClient(path=str(target_db))
        self.ef = get_embedding_function()
        self.collection = self.client.get_collection("company_docs", embedding_function=self.ef)
        
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
