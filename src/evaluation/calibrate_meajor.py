
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_benchmark" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

METRICS.mkdir(parents=True, exist_ok=True)

model = joblib.load(
    MODELS / "meajor_tfidf_lr.joblib"
)

validation = pd.read_parquet(
    DATA / "meajor_validation.parquet"
)
test = pd.read_parquet(
    DATA / "meajor_test.parquet"
)

print("=== CYB-5 THRESHOLD CALIBRATION ===")

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

y_val = validation["label"].astype(int).to_numpy()
val_probs = model.predict_proba(validation["text"])[:, 1]

# Search thresholds ONLY on TREC-06 validation.
thresholds = np.round(np.arange(0.05, 1.00, 0.01), 2)

rows = [
    calculate_metrics(y_val, val_probs, threshold)
    for threshold in thresholds
]

table = pd.DataFrame(rows)

# Target at most 5% false positives on validation.
MAX_FPR = 0.05

eligible = table.loc[table["fpr"] <= MAX_FPR]

if eligible.empty:
    raise ValueError("No threshold meets the FPR target")

# Maximize recall while meeting the FPR constraint.
best = eligible.sort_values(
    ["recall", "fpr", "threshold"],
    ascending=[False, True, False]
).iloc[0]

selected_threshold = float(best["threshold"])

print("\n=== VALIDATION THRESHOLD SELECTION ===")
print("Target maximum FPR:", MAX_FPR)
print("Selected threshold:", selected_threshold)
print(best.to_string())

print("\n=== TREC-07 EVALUATION ===")

y_test = test["label"].astype(int).to_numpy()
test_probs = model.predict_proba(test["text"])[:, 1]

default_result = calculate_metrics(
    y_test, test_probs, 0.5
)

selected_result = calculate_metrics(
    y_test, test_probs, selected_threshold
)

comparison = pd.DataFrame([
    {"setting": "Default", **default_result},
    {"setting": "Validation-selected", **selected_result}
])

print(comparison.round(4).to_string(index=False))

table.to_csv(
    METRICS / "meajor_validation_thresholds.csv",
    index=False
)

comparison.to_csv(
    METRICS / "meajor_calibration_comparison.csv",
    index=False
)

summary = {
    "validation_fpr_target": MAX_FPR,
    "selected_threshold": selected_threshold,
    "test_default": default_result,
    "test_selected": selected_result,
    "note": (
        "Threshold chosen on TREC-06 only. "
        "TREC-07 has already been inspected and "
        "is an exploratory test, not a pristine benchmark."
    )
}

(METRICS / "meajor_calibration.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== CALIBRATION COMPLETED ===")
