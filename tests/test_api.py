import os
# Safeguard environment before importing gateway or pipeline modules
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import asyncio
import subprocess
import sys
from pathlib import Path
import httpx
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import src.main as main_module
from src.main import app, get_engine, health, route_query_endpoint, run_query_endpoint, RouteRequest, QueryRequest

class MockEngine:
    """Lightweight mock engine for isolated testing without loading heavy ML models."""
    def __init__(self):
        self.call_count = 0

    def route_query(self, query: str):
        self.call_count += 1
        return {
            "class_id": 1,
            "tier": "NARROW",
            "k": 2,
            "tier_cost_usd": 0.002,
            "confidence": 0.95,
            "router_latency_ms": 0.5
        }

    def run(self, query: str, mode: str = "adaptive", static_k: int = 5):
        self.call_count += 1
        return {
            "mode": mode,
            "query": query,
            "answer": f"Mock response for '{query}'",
            "k_used": 2,
            "input_tokens": 42,
            "router_ms": 0.5,
            "retrieval_ms": 1.0,
            "total_latency_ms": 1.5,
            "tier_cost_usd": 0.002
        }

def test_get_engine_cooldown():
    """Verify that failed construction sets a cooldown deadline with monotonic clock,
    requests during cooldown return None without invoking constructor,
    and reaching the deadline allows a successful retry that is cached."""
    saved_engine = main_module.engine
    saved_deadline = main_module._retry_deadline
    saved_monotonic = main_module.time.monotonic
    orig_cls = main_module.AdaptiveRAGEngine

    simulated_time = 1000.0
    def mock_monotonic():
        return simulated_time

    constructor_calls = 0
    def failing_constructor():
        nonlocal constructor_calls
        constructor_calls += 1
        raise RuntimeError("Simulated construction failure")

    class WorkingEngine:
        pass

    def successful_constructor():
        nonlocal constructor_calls
        constructor_calls += 1
        return WorkingEngine()

    try:
        main_module.engine = None
        main_module._retry_deadline = 0.0
        main_module.time.monotonic = mock_monotonic
        main_module.AdaptiveRAGEngine = failing_constructor

        # 1. First construction fails and sets deadline
        res1 = main_module.get_engine()
        assert res1 is None
        assert constructor_calls == 1
        expected_deadline = 1000.0 + main_module.RETRY_COOLDOWN_SECONDS
        assert main_module._retry_deadline == expected_deadline, f"Expected {expected_deadline}, got {main_module._retry_deadline}"

        # 2. Request before cooldown deadline does not invoke constructor
        simulated_time = 1001.0
        res2 = main_module.get_engine()
        assert res2 is None
        assert constructor_calls == 1, "Constructor must not be invoked during cooldown"

        # 3. Advance clock to/past deadline and succeed
        simulated_time = 1002.5
        main_module.AdaptiveRAGEngine = successful_constructor
        res3 = main_module.get_engine()
        assert isinstance(res3, WorkingEngine)
        assert constructor_calls == 2

        # 4. Subsequent request reuses cached engine without re-invoking constructor
        res4 = main_module.get_engine()
        assert res4 is res3
        assert constructor_calls == 2, "Engine must be cached and constructor not called again"

        print("  [PASS] Controllable clock cooldown test passed.", flush=True)
    finally:
        main_module.engine = saved_engine
        main_module._retry_deadline = saved_deadline
        main_module.time.monotonic = saved_monotonic
        main_module.AdaptiveRAGEngine = orig_cls

async def run_isolated_http_test():
    """Isolated HTTP test around gateway app that avoids warming src.main and substitutes get_engine."""
    saved_engine = main_module.engine
    saved_deadline = main_module._retry_deadline
    saved_get_engine = main_module.get_engine

    try:
        main_module.engine = None
        main_module._retry_deadline = 0.0

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            res_health = await client.get("/health")
            assert res_health.status_code == 200
            assert res_health.json()["engine_loaded"] is False, "Initial health check should report engine_loaded=False"

            main_module.get_engine = lambda: None
            res_fail = await client.post("/route", json={"query": "test query"})
            assert res_fail.status_code == 503, f"Expected 503 on engine failure, got {res_fail.status_code}"

            mock_inst = MockEngine()
            main_module.engine = mock_inst
            main_module.get_engine = lambda: main_module.engine

            res_health2 = await client.get("/health")
            assert res_health2.status_code == 200
            assert res_health2.json()["engine_loaded"] is True

            res_route = await client.post("/route", json={"query": "What is the EU cooling-off period?"})
            assert res_route.status_code == 200
            assert res_route.json()["class_id"] == 1

            res_query = await client.post("/query", json={"query": "What is the EU cooling-off period?", "mode": "adaptive"})
            assert res_query.status_code == 200
            assert res_query.json()["k_used"] == 2
            assert mock_inst.call_count == 2, "Engine should be reused across requests"

        print("  [PASS] Isolated HTTP gateway test passed.", flush=True)
    finally:
        main_module.engine = saved_engine
        main_module._retry_deadline = saved_deadline
        main_module.get_engine = saved_get_engine

async def run_all_tests():
    print("[TEST] 1. Initializing Adaptive RAG Engine...", flush=True)
    engine = get_engine()
    assert engine is not None, "Failed to initialize engine"
    print(f"       Engine initialized successfully with collection: {getattr(engine, 'collection_name', 'company_docs')}.", flush=True)

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

    # Query Adaptive - exercises real retrieval with k_used=2
    print("[TEST] 3. Testing Adaptive Retrieval Request (k_used > 0)...", flush=True)
    q_adapt = await run_query_endpoint(QueryRequest(query="What is the EU cooling-off period for digital goods?", mode="adaptive"))
    assert q_adapt["k_used"] == 2
    assert "answer" in q_adapt
    assert q_adapt["retrieval_ms"] >= 0
    print(f"       Query Adaptive output: k_used={q_adapt['k_used']}, tokens={q_adapt['input_tokens']}", flush=True)

    # Query Static
    q_static = await run_query_endpoint(QueryRequest(query="What is the EU cooling-off period for digital goods?", mode="static", static_k=5))
    assert q_static["k_used"] == 5
    print(f"       Query Static output: k_used={q_static['k_used']}, tokens={q_static['input_tokens']}", flush=True)

    print("\n[TEST] 4. Testing 503 fallback when engine is uninitialized...", flush=True)
    saved_engine = main_module.engine
    saved_get_engine = main_module.get_engine
    try:
        main_module.engine = None
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
        main_module.get_engine = saved_get_engine

    print("\n[TEST] 5. Testing ASGI HTTP transport via httpx.AsyncClient...", flush=True)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.get("/health")
        assert res.status_code == 200

        res_route = await client.post("/route", json={"query": "What is 25 * 48?"})
        assert res_route.status_code == 200
        assert res_route.json()["class_id"] == 0

        res_query = await client.post("/query", json={"query": "What is 25 * 48?", "mode": "adaptive"})
        assert res_query.status_code == 200
        assert res_query.json()["k_used"] == 0

    print("\n[TEST] 6. Isolated HTTP gateway testing...", flush=True)
    await run_isolated_http_test()

    print("\n[TEST] 7. Controllable clock cooldown testing...", flush=True)
    test_get_engine_cooldown()

    print("\n[SUCCESS] ALL ENGINE & FASTAPI ENDPOINTS VERIFIED AND PASSING!", flush=True)

def test_api():
    """Discoverable test wrapper running offline integration test in a clean subprocess with PYTHONPATH."""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    python_path = str(PROJECT_ROOT) + os.pathsep + str(PROJECT_ROOT / "src")
    if "PYTHONPATH" in env:
        python_path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    cmd = [sys.executable, str(Path(__file__).resolve()), "--direct"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"Subprocess integration test failed with exit code {res.returncode}")
    print(res.stdout)

def test_isolated_gateway_http():
    """Discoverable isolated test for pytest."""
    asyncio.run(run_isolated_http_test())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        asyncio.run(run_all_tests())
    else:
        test_api()
