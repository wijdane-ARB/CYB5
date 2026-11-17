
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_benchmark" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

test = pd.read_parquet(DATA / "meajor_test.parquet")

lr = joblib.load(MODELS / "meajor_tfidf_lr.joblib")
nb = joblib.load(MODELS / "meajor_complement_nb.joblib")

y = test["label"].astype(int).to_numpy()

# Thresholds selected previously on TREC-06.
lr_pred = (
    lr.predict_proba(test["text"])[:, 1] >= 0.69
).astype(int)

nb_pred = (
    nb.predict_proba(test["text"])[:, 1] >= 0.64
).astype(int)

def metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(actual, predicted, zero_division=0),
        "fpr": fp / (fp + tn)
    }

lr_metrics = metrics(y, lr_pred)
nb_metrics = metrics(y, nb_pred)

print("=== CYB-5 PAIRED MODEL COMPARISON ===")

print("\nMetric       LogisticRegression   ComplementNB")
for key in lr_metrics:
    print(
        f"{key:12} {lr_metrics[key]:.4f}"
        f"               {nb_metrics[key]:.4f}"
    )

# Paired, class-stratified bootstrap.
rng = np.random.default_rng(42)
negative = np.flatnonzero(y == 0)
positive = np.flatnonzero(y == 1)

differences = {
    key: [] for key in lr_metrics
}

N_BOOTSTRAPS = 1000

print("\nRunning paired bootstrap...")

for _ in range(N_BOOTSTRAPS):
    idx = np.concatenate([
        rng.choice(
            negative, size=len(negative), replace=True
        ),
        rng.choice(
            positive, size=len(positive), replace=True
        )
    ])

    a = metrics(y[idx], lr_pred[idx])
    b = metrics(y[idx], nb_pred[idx])

    for key in differences:
        differences[key].append(b[key] - a[key])

print("\n=== DIFFERENCE: NB MINUS LR ===")

summary = {}

for key, values in differences.items():
    low, high = np.percentile(values, [2.5, 97.5])
    difference = nb_metrics[key] - lr_metrics[key]

    summary[key] = {
        "difference": float(difference),
        "ci_low": float(low),
        "ci_high": float(high)
    }

    print(
        f"{key:12} "
        f"diff={difference:+.4f} "
        f"95% CI=[{low:+.4f}, {high:+.4f}]"
    )

output = METRICS / "meajor_model_comparison.json"

output.write_text(
    json.dumps({
        "bootstrap_iterations": N_BOOTSTRAPS,
        "method": "paired stratified bootstrap",
        "lr_threshold": 0.69,
        "nb_threshold": 0.64,
        "comparison": summary
    }, indent=2),
    encoding="utf-8"
)

print("\n=== COMPARISON COMPLETED ===")
print("Saved:", output)
