from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os

try:
    from pipeline import AdaptiveRAGEngine, ACTION_MAP
except ImportError:
    from src.pipeline import AdaptiveRAGEngine, ACTION_MAP

app = FastAPI(
    title="Adaptive RAG Gateway API",
    description="Adaptive Retrieval-Augmented Generation routing gateway based on query complexity",
    version="1.0.0"
)

# Global engine instance
engine: Optional[AdaptiveRAGEngine] = None

@app.on_event("startup")
def startup_event():
    global engine
    try:
        engine = AdaptiveRAGEngine()
        print("[INFO] Adaptive RAG Engine successfully initialized.")
    except Exception as e:
        print(f"[WARNING] Adaptive RAG Engine not initialized yet: {e}")

class RouteRequest(BaseModel):
    query: str = Field(..., example="What is the EU cooling-off period for digital goods?")

class QueryRequest(BaseModel):
    query: str = Field(..., example="Compare churn rates across Q1, Q2, and Q3.")
    mode: Literal["adaptive", "static"] = "adaptive"
    static_k: int = Field(default=5, ge=0, le=20)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "action_tiers": ACTION_MAP
    }

@app.post("/route")
def route_query_endpoint(req: RouteRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models and vector store exist.")
    return engine.route_query(req.query)

@app.post("/query")
def run_query_endpoint(req: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized. Ensure models and vector store exist.")
    return engine.run(query=req.query, mode=req.mode, static_k=req.static_k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
