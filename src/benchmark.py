from typing import List, Dict, Any
import pandas as pd

try:
    from pipeline import AdaptiveRAGEngine
except ImportError:
    from src.pipeline import AdaptiveRAGEngine

TEST_QUERIES: List[str] = [
    "What is 25 * 48?",
    "Define idempotency in REST APIs.",
    "What is the EU cooling-off period for digital goods?",
    "What encryption standard is used for European data residency?",
    "Compare churn rates across Q1, Q2, and Q3 and explain why each shift happened.",
    "List all refund conditions across US, EU, and Enterprise tiers."
]

def run_benchmark() -> pd.DataFrame:
    engine = AdaptiveRAGEngine()
    records: List[Dict[str, Any]] = []

    print("[INFO] Running comparative benchmarks across test queries...")
    for q in TEST_QUERIES:
        static_res = engine.run(q, mode="static", static_k=5)
        adaptive_res = engine.run(q, mode="adaptive")
        
        token_saved = round((1 - adaptive_res["input_tokens"] / static_res["input_tokens"]) * 100, 1) if static_res["input_tokens"] > 0 else 0.0
        
        records.append({
            "query": q[:35] + "..." if len(q) > 35 else q,
            "adapt_k": adaptive_res["k_used"],
            "static_tok": static_res["input_tokens"],
            "adapt_tok": adaptive_res["input_tokens"],
            "token_saved_%": token_saved,
            "static_ms": static_res["total_latency_ms"],
            "adapt_ms": adaptive_res["total_latency_ms"],
            "router_ms": adaptive_res["router_ms"]
        })

    df = pd.DataFrame(records)
    print("\n" + "="*80)
    print("ADAPTIVE RAG BENCHMARK RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"Average Input Token Reduction: {df['token_saved_%'].mean():.1f}%\n")
    return df

if __name__ == "__main__":
    run_benchmark()
