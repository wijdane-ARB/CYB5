
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

model = joblib.load(
    ROOT / "models" / "baseline_tfidf_lr.joblib"
)

df = pd.read_csv(
    ROOT / "data" / "processed" / "nazario_external.csv"
).dropna(subset=["text"])

pred = model.predict(df["text"])

print("\n=== EXTERNAL FAILURE ANALYSIS ===")
print("Total samples:", len(df))

print("\nPredicted classes:")
print(pd.Series(pred).value_counts().sort_index())

print("\nPrediction percentages:")
print(
    pd.Series(pred).value_counts(
        normalize=True
    ).sort_index().mul(100).round(2)
)

print("\nMissed phishing emails:", int((pred == 0).sum()))
print("Detected phishing emails:", int((pred == 1).sum()))

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(df["text"])[:, 1]

    print("\nPositive-class probability summary:")
    print(pd.Series(probabilities).describe().round(4))

report = {
    "dataset": "Nazario",
    "total": int(len(df)),
    "detected": int((pred == 1).sum()),
    "missed": int((pred == 0).sum()),
    "recall": float(np.mean(pred == 1))
}

output = ROOT / "results" / "metrics" / "external_analysis.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\nReport saved:", output)
