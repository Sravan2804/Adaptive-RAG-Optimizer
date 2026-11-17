import asyncio
import threading
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from pipeline import AdaptiveRAGEngine, ACTION_MAP
except ImportError:
    from src.pipeline import AdaptiveRAGEngine, ACTION_MAP

engine: Optional[AdaptiveRAGEngine] = None
_engine_lock = threading.Lock()

def get_engine() -> Optional[AdaptiveRAGEngine]:
    global engine
    if engine is None:
        with _engine_lock:
            if engine is None:
                try:
                    engine = AdaptiveRAGEngine()
                except Exception:
                    engine = None
                    return None
    return engine

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

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "action_tiers": ACTION_MAP
    }

@app.post("/route")
async def route_query_endpoint(req: RouteRequest):
    eng = await asyncio.to_thread(get_engine)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models and vector store exist.")
    return await asyncio.to_thread(eng.route_query, req.query)

@app.post("/query")
async def run_query_endpoint(req: QueryRequest):
    eng = await asyncio.to_thread(get_engine)
    if eng is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models and vector store exist.")
    return await asyncio.to_thread(eng.run, query=req.query, mode=req.mode, static_k=req.static_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
