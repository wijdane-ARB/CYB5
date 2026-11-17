
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

model = joblib.load(
    ROOT / "models" / "baseline_tfidf_lr.joblib"
)

# Reconstruct the saved CEAS test split using hashes.
split = pd.read_csv(
    METRICS / "test_split_ids.csv"
)

ceas = pd.read_csv(DATA / "ceas_clean.csv")

test = ceas.merge(
    split[["text_hash"]].drop_duplicates(),
    on="text_hash",
    how="inner"
)

# Check the reconstructed test set.
if len(test) != len(split):
    raise ValueError(
        "Test reconstruction mismatch. "
        "Do not continue until the split is verified."
    )

nazario = pd.read_csv(
    DATA / "nazario_external.csv"
)

ceas_probs = model.predict_proba(test["text"])[:, 1]
nazario_probs = model.predict_proba(nazario["text"])[:, 1]

y_test = test["label"].astype(int).to_numpy()
y_nazario = nazario["label"].astype(int).to_numpy()

thresholds = np.arange(0.10, 0.91, 0.05)
results = []

for threshold in thresholds:
    ceas_pred = (ceas_probs >= threshold).astype(int)
    nazario_pred = (nazario_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, ceas_pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if fp + tn else 0.0

    results.append({
        "threshold": round(float(threshold), 2),
        "ceas_precision": precision_score(
            y_test, ceas_pred, zero_division=0
        ),
        "ceas_recall": recall_score(
            y_test, ceas_pred, zero_division=0
        ),
        "ceas_f1": f1_score(
            y_test, ceas_pred, zero_division=0
        ),
        "ceas_fpr": fpr,
        "nazario_recall": recall_score(
            y_nazario, nazario_pred, zero_division=0
        )
    })

df = pd.DataFrame(results)

print("\n=== THRESHOLD SENSITIVITY ANALYSIS ===")
print(df.round(4).to_string(index=False))

df.to_csv(
    METRICS / "threshold_analysis.csv",
    index=False
)

plt.figure(figsize=(9, 5))
plt.plot(
    df["threshold"],
    df["nazario_recall"],
    marker="o",
    label="Nazario phishing recall"
)
plt.plot(
    df["threshold"],
    df["ceas_fpr"],
    marker="s",
    label="CEAS false-positive rate"
)
plt.plot(
    df["threshold"],
    df["ceas_f1"],
    marker="^",
    label="CEAS F1"
)

plt.xlabel("Decision threshold")
plt.ylabel("Metric value")
plt.title("CYB-5: Threshold sensitivity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "threshold_sensitivity.png",
    dpi=200
)
plt.close()

print("\nAnalysis saved successfully.")
print("Metrics:", METRICS / "threshold_analysis.csv")
print("Graph:", FIGURES / "threshold_sensitivity.png")
