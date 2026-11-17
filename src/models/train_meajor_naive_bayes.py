
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_benchmark" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

METRICS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 COMPLEMENT NAIVE BAYES ===")

train = pd.read_parquet(DATA / "meajor_train.parquet")
validation = pd.read_parquet(
    DATA / "meajor_validation.parquet"
)
test = pd.read_parquet(DATA / "meajor_test.parquet")

for name, df, source in [
    ("Training", train, "trec5"),
    ("Validation", validation, "trec6"),
    ("Testing", test, "trec7")
]:
    assert df["source"].eq(source).all()
    assert df["label"].isin([0, 1]).all()
    print(f"{name}: {len(df)} samples")

# Verify source partitions have no exact text overlap.
for a, b in [
    (train, validation),
    (train, test),
    (validation, test)
]:
    assert set(a["text_hash"]).isdisjoint(b["text_hash"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", ComplementNB(alpha=1.0))
])

print("\nTraining ComplementNB on TREC-05...")

model.fit(
    train["text"],
    train["label"].astype(int)
)

print("Training completed!")

def evaluate(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(
            y_true, pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, pred, zero_division=0
        )),
        "fpr": float(fp / (fp + tn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

y_val = validation["label"].astype(int).to_numpy()
val_probs = model.predict_proba(
    validation["text"]
)[:, 1]

# Choose a threshold using VALIDATION ONLY.
thresholds = np.round(
    np.arange(0.01, 1.00, 0.01), 2
)

validation_results = [
    evaluate(y_val, val_probs, t)
    for t in thresholds
]

eligible = [
    row for row in validation_results
    if row["fpr"] <= 0.05
]

if not eligible:
    raise ValueError(
        "No threshold meets 5% validation FPR"
    )

best = sorted(
    eligible,
    key=lambda row: (
        -row["recall"],
        row["fpr"],
        -row["threshold"]
    )
)[0]

selected_threshold = best["threshold"]

print("\n=== VALIDATION RESULTS ===")
print("Selected threshold:", selected_threshold)
print(pd.DataFrame([best]).round(4).to_string(index=False))

# Evaluate fixed thresholds on TREC-07.
y_test = test["label"].astype(int).to_numpy()
test_probs = model.predict_proba(test["text"])[:, 1]

default = evaluate(y_test, test_probs, 0.50)
selected = evaluate(
    y_test, test_probs, selected_threshold
)

comparison = pd.DataFrame([
    {"setting": "Default", **default},
    {"setting": "Validation-selected", **selected}
])

print("\n=== TREC-07 TEST RESULTS ===")
print(comparison.round(4).to_string(index=False))

results = {
    "model": "TF-IDF + ComplementNB",
    "training_source": "trec5",
    "validation_source": "trec6",
    "test_source": "trec7",
    "validation_fpr_target": 0.05,
    "validation_selected": best,
    "test_default": default,
    "test_selected": selected,
    "limitations": [
        "Positive labels may contain generic spam",
        "TREC-07 is an exploratory evaluation set"
    ]
}

output = METRICS / "meajor_naive_bayes_results.json"

output.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

joblib.dump(
    model,
    MODELS / "meajor_complement_nb.joblib"
)

print("\n=== EXPERIMENT 19 COMPLETED ===")
print("Results saved:", output)
