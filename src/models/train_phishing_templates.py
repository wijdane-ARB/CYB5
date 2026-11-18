
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_validation" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

MODELS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 PHISHING TEMPLATE BASELINE ===")

# Load the existing template-separated partitions.
train = pd.read_parquet(DATA / "train.parquet")
validation = pd.read_parquet(DATA / "validation.parquet")
test = pd.read_parquet(DATA / "test.parquet")

for name, df in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    if df["text"].isna().any():
        raise ValueError(f"{name}: missing text")

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"{name}: both labels required")

    print(
        f"{name}: {len(df)} emails, "
        f"{df['template_id'].nunique()} templates"
    )

# Ensure that no template is shared.
for a, b in [
    (train, validation),
    (train, test),
    (validation, test)
]:
    assert set(a["template_id"]).isdisjoint(
        set(b["template_id"])
    )

    assert set(a["text_hash"]).isdisjoint(
        set(b["text_hash"])
    )

print("\nTemplate and exact-text overlap: 0")

# Build a lightweight CPU-based classifier.
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ))
])

print("\nTraining phishing classifier...")

model.fit(
    train["text"],
    train["label"].astype(int)
)

print("Training completed!")

# Select threshold ONLY using validation data.
y_val = validation["label"].astype(int).to_numpy()
val_probs = model.predict_proba(
    validation["text"]
)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)

candidates = []

for threshold in thresholds:
    prediction = (
        val_probs >= threshold
    ).astype(int)

    candidates.append({
        "threshold": float(round(threshold, 2)),
        "f1": float(f1_score(
            y_val, prediction, zero_division=0
        ))
    })

# Maximize validation F1, resolving ties
# toward the threshold closest to 0.50.
best = sorted(
    candidates,
    key=lambda r: (
        -r["f1"],
        abs(r["threshold"] - 0.50)
    )
)[0]

threshold = best["threshold"]

print("\n=== VALIDATION THRESHOLD ===")
print("Selected threshold:", threshold)
print("Validation F1:", round(best["f1"], 4))

def evaluate(name, df):
    labels = df["label"].astype(int).to_numpy()
    probabilities = model.predict_proba(
        df["text"]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    scores = {
        "samples": int(len(df)),
        "accuracy": float(accuracy_score(
            labels, predictions
        )),
        "precision": float(precision_score(
            labels, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            labels, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            labels, predictions, zero_division=0
        )),
        "fpr": float(fp / (fp + tn))
        if fp + tn else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    print(f"\n=== {name.upper()} ===")
    print(classification_report(
        labels, predictions,
        labels=[0, 1],
        zero_division=0
    ))

    print("Confusion matrix:")
    print([[int(tn), int(fp)],
           [int(fn), int(tp)]])

    print("Detailed metrics:")
    for key, value in scores.items():
        print(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    return scores

validation_scores = evaluate(
    "Template validation", validation
)

test_scores = evaluate(
    "Unseen template test", test
)

# Save the model and measured results.
result = {
    "experiment": "Phishing template baseline",
    "classifier": "TF-IDF + Logistic Regression",
    "training_samples": len(train),
    "training_templates": int(
        train["template_id"].nunique()
    ),
    "selected_threshold": threshold,
    "validation": validation_scores,
    "test": test_scores,
    "limitations": [
        "Synthetic template-generated corpus",
        "Validation threshold selected using F1",
        "Test performance applies to held-out templates",
        "Independent real-world validation still needed"
    ]
}

output = METRICS / "phishing_template_baseline.json"

output.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8"
)

joblib.dump(
    model,
    MODELS / "phishing_template_lr.joblib"
)

print("\n=== EXPERIMENT 26 COMPLETED ===")
print("Results:", output)
print("Model saved successfully.")
