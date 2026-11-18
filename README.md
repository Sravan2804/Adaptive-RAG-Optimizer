---
title: Adaptive RAG Optimizer
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Adaptive RAG Optimizer & Gateway

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?logo=database&logoColor=white)](https://www.trychroma.com/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-Router%20Classifier-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](#)

An intelligent, latency-optimized Retrieval-Augmented Generation (RAG) backend engineered to dynamically allocate retrieval chunk budgets ($k \in \{0, 2, 6\}$) based on incoming query complexity, featuring a sub-0.05ms hybrid fast-path, stateless HashingVectorizer classification, real-time analytics, and a live interactive React playground.

---

## Executive Summary

Conventional RAG pipelines query vector databases with a fixed chunk budget (e.g., $k=5$) regardless of whether a user is asking an arithmetic question, requesting a single policy date, or seeking an exhaustive multi-quarter financial synthesis.

### The Inefficiencies of Static RAG:

1. **Unnecessary Token Bloat**: Closed-book questions ("What is 25 \* 48?") do not require external context. Ingesting $k=5$ chunks into the prompt inflates token consumption by up to **80%**.
2. **Context Poisoning & Latency Penalties**: Irrelevant context chunks add unnecessary vector search latency and can distract the LLM generator.
3. **Under-Retrieval on Complex Queries**: Complex multi-hop queries often need deeper retrieval than a shallow static $k$ can supply.

### The Adaptive RAG Solution:

**Adaptive RAG Optimizer** deploys a multi-tiered routing pipeline at the ingestion gateway:

- **Tier 0 (Deterministic Fast-Path)**: Sub-0.05ms regex classification for arithmetic calculations and standalone greetings.
- **Tier 1 (`SKIP`, $k=0$)**: Bypass vector search completely for closed-book, reasoning, or chit-chat queries.
- **Tier 2 (`NARROW`, $k=2$)**: Retrieve minimal targeted context for direct factual lookups.
- **Tier 3 (`DEEP`, $k=6$)**: Retrieve broader context for synthesis, comparisons, and complex reasoning.
- **Uncertainty Guardrail**: Automatically promotes predictions with low classification confidence ($< 0.45$) to `NARROW` ($k=2$) to guarantee answer safety.

---

## Classifier Selection & Architecture Deep-Dive

### What Classifier Did We Use and Why?

The query router uses an end-to-end `scikit-learn` pipeline:

```python
Pipeline([
    ("hasher", HashingVectorizer(n_features=4096, alternate_sign=False, ngram_range=(1, 2), norm=None)),
    ("tfidf", TfidfTransformer(sublinear_tf=True)),
    ("clf", LogisticRegression(C=2.0, max_iter=1000))
])
```

#### 1. Why Logistic Regression Instead of a Heavy Transformer (BERT / MiniLM)?

- **Critical-Path Latency (< 2ms vs 30-100ms)**: The query router sits directly in the synchronous path of _every_ incoming request. Using a heavy transformer router adds 30–80ms of CPU inference overhead, defeating the purpose of latency reduction on simple queries. Logistic Regression executes in **~1.8 milliseconds**.
- **Calibrated Softmax Probabilities**: Logistic Regression natively outputs smooth, well-calibrated posterior probabilities $\sigma(W x + b)$. This is **essential for our Uncertainty Guardrail**: when $\max P(y \mid x) < 0.45$, the gateway automatically falls back to `NARROW` ($k=2$). In contrast, Naive Bayes exhibits extreme probability polarization (pushing confidences artificially to $0.0$ or $1.0$), while SVMs require expensive Platt scaling.
- **Zero GPU / Minimal Runtime**: The trained artifact is **~132 KB** on disk. It requires no GPU, no CUDA drivers, and incurs zero cold-start delay.

#### 2. Why `HashingVectorizer` Replaced the String Dictionary?

- In standard `TfidfVectorizer`, an in-memory string vocabulary dictionary is stored and serialized with the model.
- By using `HashingVectorizer(norm=None)` with 4,096 feature buckets and bi-grams (`ngram_range=(1, 2)`):
  - **Stateless & OOV Resilient**: The feature mapping uses MurmurHash3, completely eliminating the vocabulary dictionary from memory.
  - **Zero Vocabulary Memory Bloat**: Fixed memory footprint regardless of dataset growth.
  - `norm=None` passes raw bucket counts to `TfidfTransformer(sublinear_tf=True)`, which applies IDF weighting, sublinear term-frequency dampening ($1 + \log(\text{tf})$), and L2 normalization.

#### 3. Multi-Tiered Hybrid Fast-Path (0.018ms Benchmark Latency)

Before executing vectorization and classifier inference, queries pass through pre-compiled regex filters that match arithmetic expressions and standalone greetings. Unmatched queries (such as unit conversions or questions with topics) bypass the fast-path and proceed to the ML classifier; if the classifier chooses Class 0, the query returns `SKIP` with `fast_path: false`.

- In benchmark measurements under local test conditions, fast-path queries achieved a router decision latency of **0.018 ms** (a **99% reduction in routing decision latency** compared to the ~1.8 ms ML classifier).

---

## Architecture & Query Lifecycle

```mermaid
flowchart TD
    UserQuery["Incoming User Request"] --> Endpoint{"Target Endpoint"}

    %% /route Flow
    Endpoint -- "POST /route" --> RouteFastPath{"Fast-Path Regex Match?<br/>(Math, Standalone Greetings)"}
    RouteFastPath -- "Yes" --> RouteFastResp["Immediate Response (k=0, SKIP, fast_path: true)<br/>No retrieval, no generation"]
    RouteFastPath -- "No" --> RouteML["Hashing ML Router + Guardrail"]
    RouteML --> RouteMLResp["Immediate Response (class_id, tier, k, confidence)<br/>No retrieval, no generation"]

    %% /query Flow
    Endpoint -- "POST /query" --> ModeCheck{"Execution Mode"}

    %% Static Mode
    ModeCheck -- "mode = 'static'" --> StaticRetrieval["Bypass Router: Retrieve static_k Chunks<br/>from ChromaDB Vector Store"]
    StaticRetrieval --> StaticPrompt["Assemble Contextual Prompt"]
    StaticPrompt --> StaticGen["LLM Generator / Synthesizer"]
    StaticGen --> StaticResp["Return Final Answer + Static Metrics"]

    %% Adaptive Mode
    ModeCheck -- "mode = 'adaptive'" --> AdaptFastPath{"Fast-Path Regex Match?<br/>(Math, Standalone Greetings)"}
    AdaptFastPath -- "Yes" --> AdaptSkip["SKIP Tier (k=0): Zero Vector Retrieval"]

    AdaptFastPath -- "No" --> AdaptML["Hashing ML Router (4096 Buckets)"]
    AdaptML --> AdaptConf{"Confidence >= 0.45?"}
    AdaptConf -- "No (Uncertain)" --> AdaptFallback["Safety Guardrail: Fallback to NARROW (k=2)"]
    AdaptConf -- "Yes" --> AdaptTier{"Predicted Tier"}

    AdaptTier -- "Class 0" --> AdaptSkip
    AdaptTier -- "Class 1" --> AdaptNarrow["NARROW Tier (k=2): Factual Retrieval"]
    AdaptTier -- "Class 2" --> AdaptDeep["DEEP Tier (k=6): Multi-Hop Retrieval"]

    AdaptFallback --> AdaptChroma[("ChromaDB Vector Store")]
    AdaptNarrow --> AdaptChroma
    AdaptDeep --> AdaptChroma

    AdaptSkip --> AdaptPrompt["Assemble Prompt (Context if any)"]
    AdaptChroma --> AdaptPrompt
    AdaptPrompt --> AdaptGen["LLM Generator / Synthesizer"]
    AdaptGen --> AdaptResp["Return Final Answer + Token/Latency Metrics"]
```

---

## Interactive Playground Dashboard

The project includes a self-contained, offline-ready single-page dashboard served directly by FastAPI at **`http://localhost:8000/dashboard`** (or `/`):

- **Real-Time KPI Cards**: Live counters for Total Queries, Fast-Path Ratio, Cumulative Tokens Saved (%), and Estimated Dollar Savings.
- **Interactive Playground**: Test queries with one-click presets or custom prompts across Adaptive and Static modes.
- **Dynamic Tier Indicators**: Real-time glow badges (`SKIP` in Emerald, `NARROW` in Sky-Blue, `DEEP` in Purple) with fast-path detection pills.
- **Latency Waterfall**: Router decision latency, vector retrieval latency, and end-to-end response times.
- **Context Document Accordion**: Live inspection of retrieved ChromaDB chunks.
- **Live Activity Stream**: Real-time tabular feed of recent queries displaying input tokens, latencies, tiers, and routing mechanism.

---

## Action Tiers & Cost Model

| Tier         | Class ID | Chunk Budget ($k$) | Intended Query Complexity              | Avg. Input Tokens | Tier Cost (USD) | Example Query                                                                      |
| ------------ | :------: | :----------------: | -------------------------------------- | :---------------: | :-------------: | ---------------------------------------------------------------------------------- |
| **`SKIP`**   |   `0`    |        $0$         | Closed-book, arithmetic, definitions   |        ~27        |    \$0.0005     | _"What is 25 _ 48?"\*                                                              |
| **`NARROW`** |   `1`    |        $2$         | Direct factual policy / product lookup |        ~83        |    \$0.0020     | _"What is the EU cooling-off period for digital goods?"_                           |
| **`DEEP`**   |   `2`    |        $6$         | Cross-document analysis, synthesis     |       ~200        |    \$0.0060     | _"Compare churn rates across Q1, Q2, and Q3 and explain why each shift happened."_ |

---

## Empirical Benchmark Results

Running `src/benchmark.py` directly compares **Adaptive RAG** against a conventional **Static ($k=5$) baseline**:

```text
================================================================================
ADAPTIVE RAG BENCHMARK RESULTS
================================================================================
                                 query  adapt_k  static_tok  adapt_tok  token_saved_%  static_ms  adapt_ms  router_ms
                      What is 25 * 48?        0         167         27           83.8     113.04      0.08      0.018
      Define idempotency in REST APIs.        0         163         27           83.4      27.56    101.54    101.430
What is the EU cooling-off period f...        2         163         83           49.1      24.60     24.46      2.650
What encryption standard is used fo...        2         168         88           47.6      21.50     23.16      2.770
Compare churn rates across Q1, Q2, ...        6         184        210          -14.1      23.25     23.35      2.400
List all refund conditions across U...        6         168        197          -17.3      22.87     29.34      2.500
================================================================================
Average Input Token Reduction: 38.7%
```

> [!NOTE]
> Latency figures represent benchmark measurements under local test conditions rather than runtime guarantees. For fast-path queries (_"What is 25 _ 48?"\*), router decision latency measured **0.018 ms** (a 99% reduction vs the ~1.8 ms ML classifier), and end-to-end query latency measured **0.08 ms** using local direct/synthetic generation output (excluding external LLM API network transport), achieving an **83.8% input token reduction** with zero vector retrieval.

---

## Project Structure

```
Adaptive-RAG-Optimizer/
├── data/
│   ├── knowledge_base.json         # Knowledge base documents (policies, SLA, security)
│   └── router_dataset.csv          # Labeled queries dataset (query, label)
├── models/
│   └── query_router.joblib         # 132 KB stateless HashingVectorizer classifier
├── src/
│   ├── __init__.py
│   ├── ingest.py                   # Vector store ingestion with dual-backend & atomic tags
│   ├── train_router.py             # Router training with HashingVectorizer + LogisticRegression
│   ├── pipeline.py                 # Hybrid fast-path & AdaptiveRAGEngine core logic
│   ├── benchmark.py                # Comparative benchmark suite
│   ├── main.py                     # FastAPI REST API & live AnalyticsTracker
│   └── static/
│       └── index.html              # Live React 18 Single-Page Dashboard
├── tests/
│   └── test_api.py                 # Comprehensive unit, ASGI & cooldown test suite
├── scripts/
│   └── audit_code.py               # Custom AST-based static code auditor
├── chroma_db/                      # Persistent ChromaDB vector store
├── requirements.txt                # Python package dependencies
├── .env                            # Environment variables (offline flags, keys)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and prepare a Python 3.10+ virtual environment:

```bash
git clone https://github.com/your-username/Adaptive-RAG-Optimizer.git
cd Adaptive-RAG-Optimizer

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)

Create or update `.env` in the project root:

```env
OPENAI_API_KEY=${OPENAI_API_KEY}
HF_HUB_OFFLINE=1
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

## Execution Guide

### Step 1: Train the Complexity Router

Train the stateless HashingVectorizer and Logistic Regression classifier:

```bash
python src/train_router.py
```

_Output: Saves the trained model (~132 KB) to `models/query_router.joblib`._

### Step 2: Ingest Documents into Vector Store

Ingest `data/knowledge_base.json` into ChromaDB:

```bash
python src/ingest.py
```

### Step 3: Run Comparative Benchmarks

Evaluate latency and token savings against the baseline:

```bash
python src/benchmark.py
```

### Step 4: Run the Test Suite

Execute the comprehensive offline-safe test suite:

```bash
python tests/test_api.py --direct
```

### Step 5: Launch the Server & Open Dashboard

Start the web service:

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive React Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard) (or [http://localhost:8000](http://localhost:8000))
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Reference

### 1. Real-Time Analytics Endpoint

**`GET /analytics`**

Returns live cumulative token metrics, tier distribution, fast-path ratio, and estimated cost savings.

```bash
curl -X GET http://127.0.0.1:8000/analytics
```

**Response (200 OK):**

```json
{
  "total_queries": 42,
  "fast_path_hits": 18,
  "fast_path_ratio_pct": 42.8,
  "tier_distribution": {
    "SKIP": {"count": 18, "pct": 42.8},
    "NARROW": {"count": 16, "pct": 38.1},
    "DEEP": {"count": 8, "pct": 19.1}
  },
  "token_metrics": {
    "total_tokens_used": 2840,
    "total_static_baseline_tokens": 6800,
    "total_tokens_saved": 3960,
    "token_savings_pct": 58.2
  },
  "cost_metrics_usd": {
    "total_cost_usd": 0.082,
    "static_baseline_cost_usd": 0.21,
    "estimated_usd_saved": 0.128,
    "cost_savings_pct": 60.9
  },
  "performance_ms": {
    "avg_router_latency_ms": 1.4,
    "avg_retrieval_latency_ms": 28.5,
    "avg_total_latency_ms": 32.1
  },
  "recent_queries": [...]
}
```

---

### 2. Route Query Endpoint

**`POST /route`**

Classifies the query and returns the recommended tier without performing retrieval.

```bash
curl -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 25 * 48?"}'
```

**Response (200 OK - Fast-Path):**

```json
{
  "class_id": 0,
  "tier": "SKIP",
  "k": 0,
  "tier_cost_usd": 0.0005,
  "confidence": 1.0,
  "router_latency_ms": 0.018,
  "fast_path": true
}
```

---

### 3. Full Query & Retrieval Endpoint

**`POST /query`**

Routes the query, retrieves chunks if $k > 0$, and synthesizes the final response.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the EU cooling-off period for digital goods?",
    "mode": "adaptive"
  }'
```

---

## Reliability & Resilience Engineering

1. **Deterministic Fast-Path**: Handles arithmetic, mathematical queries, and greetings in $0.018\text{ ms}$, entirely skipping both the ML router and ChromaDB.
2. **Stateless HashingVectorizer**: Replaced in-memory dictionary lookups with fixed 4,096 MurmurHash3 feature buckets.
3. **Monotonic Clock Cooldown (`RETRY_COOLDOWN_SECONDS = 2.0`)**: Guarantees fast failure during transient initialization errors without cascading retry loops.
4. **Double-Checked Locking**: Thread-safe lazy initialization ensures only one worker thread initializes the models during cold starts.
5. **ChromaDB $\ge 0.6.x$ & Legacy Compatibility**: Automatically detects and handles string collection names vs older Collection objects.
6. **Atomic Revision Tagging & Stale Document Purge**: Ingests under `"building"` state, purges stale documents, verifies document counts, and atomically transitions to `"ready"` with a SHA-256 fingerprint.

---

## License

This project is licensed under the [MIT License](LICENSE).
