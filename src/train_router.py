import os
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists("data/router_dataset.csv"):
        dataset_path = "data/router_dataset.csv"
        model_dir = "models"
        model_path = "models/query_router.joblib"
    else:
        dataset_path = os.path.join(base_dir, "data", "router_dataset.csv")
        model_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(model_dir, "query_router.joblib")
    return dataset_path, model_dir, model_path

def train():
    dataset_path, model_dir, model_path = get_paths()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing training set at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    X, y = df["query"], df["label"]
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=2.0, max_iter=1000))
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=3)
    print(f"[INFO] 3-Fold Cross-Validation Accuracy: {scores.mean():.2f}")
    
    pipeline.fit(X, y)
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    # Latency benchmarking check
    sample = ["Compare enterprise churn trends across Q1 and Q3."]
    start = time.perf_counter()
    pred = pipeline.predict(sample)[0]
    probs = pipeline.predict_proba(sample)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    
    print(f"[SUCCESS] Router saved to {model_path}. Sample prediction: class={pred}, confidence={max(probs):.2f}, latency={latency_ms:.2f}ms")

if __name__ == "__main__":
    train()
