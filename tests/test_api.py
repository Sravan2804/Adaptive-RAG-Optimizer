import os
import sys
import asyncio
from pathlib import Path
import httpx
from fastapi import HTTPException

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.main as main_module
from src.main import app, get_engine, health, route_query_endpoint, run_query_endpoint, RouteRequest, QueryRequest

async def run_all_tests():
    print("[TEST] 1. Initializing Adaptive RAG Engine...", flush=True)
    engine = get_engine()
    assert engine is not None, "Failed to initialize engine"
    print("       Engine initialized successfully.", flush=True)

    print("[TEST] 2. Direct FastAPI route handler testing (async + to_thread)...", flush=True)
    h_res = await health()
    assert h_res["status"] == "healthy"
    assert h_res["engine_loaded"] is True
    print(f"       Health endpoint output: {h_res}", flush=True)

    # Route Class 0 (SKIP)
    r0 = await route_query_endpoint(RouteRequest(query="What is 25 * 48?"))
    assert r0["class_id"] == 0
    assert r0["tier"] == "SKIP"
    assert r0["k"] == 0
    print(f"       Route Class 0 (SKIP): {r0}", flush=True)

    # Route Class 1 (NARROW)
    r1 = await route_query_endpoint(RouteRequest(query="What is the EU cooling-off period for digital goods?"))
    assert r1["class_id"] == 1
    assert r1["tier"] == "NARROW"
    assert r1["k"] == 2
    print(f"       Route Class 1 (NARROW): {r1}", flush=True)

    # Route Class 2 (DEEP)
    r2 = await route_query_endpoint(RouteRequest(query="Compare churn rates across Q1, Q2, and Q3 and explain why each shift happened."))
    assert r2["class_id"] == 2
    assert r2["tier"] == "DEEP"
    assert r2["k"] == 6
    print(f"       Route Class 2 (DEEP): {r2}", flush=True)

    # Query Adaptive
    q_adapt = await run_query_endpoint(QueryRequest(query="What is the EU cooling-off period for digital goods?", mode="adaptive"))
    assert q_adapt["k_used"] == 2
    assert "answer" in q_adapt
    print(f"       Query Adaptive output: k_used={q_adapt['k_used']}, tokens={q_adapt['input_tokens']}", flush=True)

    # Query Static
    q_static = await run_query_endpoint(QueryRequest(query="What is the EU cooling-off period for digital goods?", mode="static", static_k=5))
    assert q_static["k_used"] == 5
    print(f"       Query Static output: k_used={q_static['k_used']}, tokens={q_static['input_tokens']}", flush=True)

    print("\n[TEST] 3. Testing 503 fallback when engine is uninitialized...", flush=True)
    saved_engine = main_module.engine
    try:
        # Simulate uninitialized / failed engine
        main_module.engine = None
        # Mock get_engine to return None
        orig_get_engine = main_module.get_engine
        main_module.get_engine = lambda: None
        try:
            await route_query_endpoint(RouteRequest(query="Test"))
            assert False, "Expected 503 HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 503, f"Expected 503, got {exc.status_code}"
            print(f"       503 handled correctly on /route: {exc.detail}", flush=True)

        try:
            await run_query_endpoint(QueryRequest(query="Test"))
            assert False, "Expected 503 HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 503, f"Expected 503, got {exc.status_code}"
            print(f"       503 handled correctly on /query: {exc.detail}", flush=True)
    finally:
        main_module.engine = saved_engine
        main_module.get_engine = orig_get_engine

    print("\n[TEST] 4. Testing ASGI HTTP transport via httpx.AsyncClient...", flush=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        print(f"       HTTP GET /health status: {res.status_code}, data: {res.json()}", flush=True)

        res_route = await client.post("/route", json={"query": "What is 25 * 48?"})
        assert res_route.status_code == 200
        assert res_route.json()["class_id"] == 0
        print(f"       HTTP POST /route status: {res_route.status_code}, class: {res_route.json()['class_id']}", flush=True)

        res_query = await client.post("/query", json={"query": "What is 25 * 48?", "mode": "adaptive"})
        assert res_query.status_code == 200
        assert res_query.json()["k_used"] == 0
        print(f"       HTTP POST /query status: {res_query.status_code}, k_used: {res_query.json()['k_used']}", flush=True)

    print("\n[SUCCESS] ALL ENGINE & FASTAPI ENDPOINTS VERIFIED AND PASSING!", flush=True)

def test_api():
    """Discoverable test wrapper for pytest and standard test runners."""
    asyncio.run(run_all_tests())

if __name__ == "__main__":
    asyncio.run(run_all_tests())
