import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

try:
    from pipeline import AdaptiveRAGEngine, ACTION_MAP
except ImportError:
    from src.pipeline import AdaptiveRAGEngine, ACTION_MAP

logger = logging.getLogger("adaptive_rag_gateway")

engine: Optional[AdaptiveRAGEngine] = None
_engine_lock = threading.Lock()
_retry_deadline: float = 0.0
RETRY_COOLDOWN_SECONDS: float = 2.0
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_INDEX = STATIC_DIR / "index.html"

class AnalyticsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_queries = 0
        self.total_query_requests = 0
        self.tier_counts = {0: 0, 1: 0, 2: 0}
        self.fast_path_hits = 0
        self.total_tokens_used = 0
        self.total_static_tokens = 0
        self.total_usd_cost = 0.0
        self.total_static_usd_cost = 0.0
        self.total_router_ms = 0.0
        self.total_retrieval_ms = 0.0
        self.total_latency_ms = 0.0
        self.query_history: List[Dict[str, Any]] = []

    def record_route(self, route_info: dict):
        with self._lock:
            self.total_queries += 1
            cid = route_info.get("class_id", 1)
            self.tier_counts[cid] = self.tier_counts.get(cid, 0) + 1
            if route_info.get("fast_path"):
                self.fast_path_hits += 1
            self.total_router_ms += route_info.get("router_latency_ms", 0.0)

    def record_query(self, res: dict):
        with self._lock:
            self.total_queries += 1
            self.total_query_requests += 1
            mode = res.get("mode", "adaptive")
            k_used = res.get("k_used", 0)

            # Only record into adaptive tier distribution if not static mode
            if mode == "adaptive":
                cid = 0 if k_used == 0 else (1 if k_used <= 2 else 2)
                self.tier_counts[cid] = self.tier_counts.get(cid, 0) + 1

            if res.get("fast_path"):
                self.fast_path_hits += 1

            tokens = res.get("input_tokens", 0)
            self.total_tokens_used += tokens

            # If request is reference static k=5 configuration, contribute measured tokens unchanged
            if mode == "static" and k_used == 5:
                static_tokens = tokens
            elif k_used == 0:
                static_tokens = 165
            elif k_used == 2:
                static_tokens = int(tokens * 1.5)
            else:
                static_tokens = int(tokens * 0.9)
            self.total_static_tokens += static_tokens

            cost = res.get("tier_cost_usd", 0.002)
            self.total_usd_cost += cost
            self.total_static_usd_cost += 0.005

            self.total_router_ms += res.get("router_ms", 0.0)
            self.total_retrieval_ms += res.get("retrieval_ms", 0.0)
            self.total_latency_ms += res.get("total_latency_ms", 0.0)

            # Store execution metadata without exposing raw query prompts in shared view
            item = {
                "timestamp": time.strftime("%H:%M:%S"),
                "mode": mode,
                "tier": res.get("tier", "NARROW"),
                "k_used": k_used,
                "tokens": tokens,
                "latency_ms": res.get("total_latency_ms", 0.0),
                "fast_path": res.get("fast_path", False)
            }
            self.query_history.insert(0, item)
            if len(self.query_history) > 50:
                self.query_history.pop()

    def get_snapshot(self) -> dict:
        with self._lock:
            t_queries = max(1, self.total_queries)
            t_full_queries = max(1, self.total_query_requests)
            adaptive_total = max(1, sum(self.tier_counts.values()))
            saved_tokens = max(0, self.total_static_tokens - self.total_tokens_used)
            saved_tokens_pct = round((saved_tokens / max(1, self.total_static_tokens)) * 100, 1) if self.total_static_tokens > 0 else 0.0
            saved_usd = max(0.0, self.total_static_usd_cost - self.total_usd_cost)

            return {
                "total_queries": self.total_queries,
                "total_query_requests": self.total_query_requests,
                "fast_path_hits": self.fast_path_hits,
                "fast_path_ratio_pct": round((self.fast_path_hits / t_queries) * 100, 1) if self.total_queries > 0 else 0.0,
                "tier_distribution": {
                    "SKIP": {"count": self.tier_counts.get(0, 0), "pct": round((self.tier_counts.get(0, 0) / adaptive_total) * 100, 1)},
                    "NARROW": {"count": self.tier_counts.get(1, 0), "pct": round((self.tier_counts.get(1, 0) / adaptive_total) * 100, 1)},
                    "DEEP": {"count": self.tier_counts.get(2, 0), "pct": round((self.tier_counts.get(2, 0) / adaptive_total) * 100, 1)}
                },
                "token_metrics": {
                    "total_tokens_used": self.total_tokens_used,
                    "total_static_baseline_tokens": self.total_static_tokens,
                    "total_tokens_saved": saved_tokens,
                    "token_savings_pct": saved_tokens_pct
                },
                "cost_metrics_usd": {
                    "total_cost_usd": round(self.total_usd_cost, 4),
                    "static_baseline_cost_usd": round(self.total_static_usd_cost, 4),
                    "estimated_usd_saved": round(saved_usd, 4),
                    "cost_savings_pct": round((saved_usd / max(0.001, self.total_static_usd_cost)) * 100, 1) if self.total_static_usd_cost > 0 else 0.0
                },
                "performance_ms": {
                    "avg_router_latency_ms": round(self.total_router_ms / t_queries, 2),
                    "avg_retrieval_latency_ms": round(self.total_retrieval_ms / t_full_queries, 2) if self.total_query_requests > 0 else 0.0,
                    "avg_total_latency_ms": round(self.total_latency_ms / t_full_queries, 2) if self.total_query_requests > 0 else 0.0
                },
                "recent_queries": self.query_history[:15]
            }

tracker = AnalyticsTracker()

def get_engine() -> Optional[AdaptiveRAGEngine]:
    global engine, _retry_deadline
    # Fast path: already initialized
    if engine is not None:
        return engine

    # Fast rejection during retry cooldown window
    if time.monotonic() < _retry_deadline:
        return None

    with _engine_lock:
        # Check again in case another concurrent thread initialized it
        if engine is not None:
            return engine
        # Check if another concurrent thread failed and set the cooldown
        if time.monotonic() < _retry_deadline:
            return None
        try:
            engine = AdaptiveRAGEngine()
            return engine
        except FileNotFoundError as fnf:
            logger.warning("Vector store missing on startup (%s). Auto-running ingestion...", fnf)
            try:
                try:
                    from ingest import build_vector_store
                except ImportError:
                    from src.ingest import build_vector_store
                build_vector_store()
                engine = AdaptiveRAGEngine()
                return engine
            except Exception as ingest_exc:
                _retry_deadline = time.monotonic() + RETRY_COOLDOWN_SECONDS
                engine = None
                logger.error("Failed to auto-ingest knowledge base: %s", ingest_exc)
                return None
        except Exception as exc:
            _retry_deadline = time.monotonic() + RETRY_COOLDOWN_SECONDS
            engine = None
            logger.error("Failed to initialize AdaptiveRAGEngine: %s", exc)
            return None

app = FastAPI(
    title="Adaptive RAG Gateway API",
    description="Adaptive Retrieval-Augmented Generation routing gateway based on query complexity",
    version="1.0.0"
)

class RouteRequest(BaseModel):
    query: str = Field(..., example="What is the EU cooling-off period for digital goods?")

class QueryRequest(BaseModel):
    query: str = Field(..., example="Compare churn rates across Q1, Q2, and Q3.")
    mode: Literal["adaptive", "static"] = "adaptive"
    static_k: int = Field(default=5, ge=0, le=20)

@app.get("/")
@app.get("/dashboard")
async def dashboard():
    target = STATIC_INDEX if STATIC_INDEX.exists() else Path("src/static/index.html")
    if target.exists():
        return FileResponse(str(target.resolve()), media_type="text/html")
    return HTMLResponse("<h2>Adaptive RAG Optimizer Dashboard</h2><p>Static index not found.</p>", status_code=404)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "action_tiers": ACTION_MAP
    }

@app.get("/analytics")
async def analytics():
    return tracker.get_snapshot()

@app.post("/route")
async def route_query_endpoint(req: RouteRequest):
    eng = await asyncio.to_thread(get_engine)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models and vector store exist.")
    res = await asyncio.to_thread(eng.route_query, req.query)
    tracker.record_route(res)
    return res

@app.post("/query")
async def run_query_endpoint(req: QueryRequest):
    eng = await asyncio.to_thread(get_engine)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models and vector store exist.")
    res = await asyncio.to_thread(eng.run, query=req.query, mode=req.mode, static_k=req.static_k)
    tracker.record_query(res)
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
