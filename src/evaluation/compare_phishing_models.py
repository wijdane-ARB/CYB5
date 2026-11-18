
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "phishing_validation" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

METRICS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 EXPERIMENT 27 ===")
print("COMPARISON ON UNSEEN PHISHING TEMPLATES")

test = pd.read_parquet(DATA / "test.parquet")

if set(test["label"].unique()) != {0, 1}:
    raise ValueError("Both classes are required")

if test["text_hash"].duplicated().any():
    raise ValueError("Duplicate test messages detected")

models = {
    "TREC_LogisticRegression": (
        "meajor_tfidf_lr.joblib", 0.69
    ),
    "TREC_ComplementNB": (
        "meajor_complement_nb.joblib", 0.64
    ),
    "Phishing_LogisticRegression": (
        "phishing_template_lr.joblib", 0.44
    )
}

y_true = test["label"].astype(int).to_numpy()

results = []
template_results = []

for model_name, (filename, threshold) in models.items():

    print(f"\nEvaluating: {model_name}")

    model = joblib.load(MODELS / filename)

    probabilities = model.predict_proba(
        test["text"]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "model": model_name,
        "threshold": threshold,
        "samples": len(test),
        "accuracy": accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions,
            zero_division=0
        ),
        "fpr": fp / (fp + tn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    })

    # Report performance for each unseen template.
    temp = test[["template_id", "label"]].copy()
    temp["prediction"] = predictions

    for template_id, group in temp.groupby("template_id"):

        template_results.append({
            "model": model_name,
            "template_id": str(template_id),
            "samples": len(group),
            "label": int(group["label"].iloc[0]),
            "accuracy": accuracy_score(
                group["label"],
                group["prediction"]
            )
        })

summary = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")
print(
    summary.round(4).to_string(index=False)
)

templates = pd.DataFrame(template_results)

print("\n=== PERFORMANCE BY TEMPLATE ===")
print(
    templates.round(4).to_string(index=False)
)

summary.to_csv(
    METRICS / "phishing_model_comparison.csv",
    index=False
)

templates.to_csv(
    METRICS / "phishing_template_performance.csv",
    index=False
)

(METRICS / "phishing_model_comparison.json").write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\n=== EXPERIMENT 27 COMPLETED ===")
print("Comparison and template results saved.")
