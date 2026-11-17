
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

model = joblib.load(
    ROOT / "models" / "baseline_tfidf_lr.joblib"
)

df = pd.read_csv(
    DATA / "nazario_external.csv"
).dropna(subset=["text", "label"])

texts = df["text"].astype(str)

footers = {
    "original": "",
    "neutral_footer": (
        "\n\nThis message is intended for its recipient."
    ),
    "short_neutral": "\n\nThank you.",
    "long_neutral": (
        "\n\nThis message is intended for its recipient. "
        "Thank you for your attention. "
        "Please retain this message for your records."
    ),
    "repeated_neutral": (
        "\n\nThis message is intended for its recipient."
        * 3
    ),
    "numeric_padding": "\n\n12345 67890 24680 13579"
}

base_predictions = model.predict(texts)
base_probabilities = model.predict_proba(texts)[:, 1]

rows = []

print("\n=== CYB-5 FOOTER MECHANISM ANALYSIS ===")
print("Total phishing emails:", len(df))

for name, footer in footers.items():
    modified = texts + footer

    predictions = model.predict(modified)
    probabilities = model.predict_proba(modified)[:, 1]

    newly_missed = int((
        (base_predictions == 1) &
        (predictions == 0)
    ).sum())

    recovered = int((
        (base_predictions == 0) &
        (predictions == 1)
    ).sum())

    rows.append({
        "variant": name,
        "footer_characters": len(footer),
        "detected": int((predictions == 1).sum()),
        "missed": int((predictions == 0).sum()),
        "recall": float((predictions == 1).mean()),
        "newly_missed": newly_missed,
        "recovered": recovered,
        "mean_probability": float(
            probabilities.mean()
        ),
        "mean_probability_change": float(
            (probabilities - base_probabilities).mean()
        )
    })

results = pd.DataFrame(rows)

print("\n=== RESULTS ===")
print(results.round(4).to_string(index=False))

results.to_csv(
    METRICS / "footer_analysis.csv",
    index=False
)

print("\nResults saved successfully.")
