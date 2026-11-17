
from pathlib import Path
import json
import re

import joblib
import pandas as pd
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

model = joblib.load(
    ROOT / "models" / "baseline_tfidf_lr.joblib"
)

df = pd.read_csv(
    DATA / "nazario_external.csv"
).dropna(subset=["text", "label"])

assert set(df["label"].unique()) == {1}, (
    "Expected positive-only Nazario corpus"
)

def uppercase(text):
    return str(text).upper()

def extra_spaces(text):
    return re.sub(r" +", "   ", str(text))

def punctuation_removed(text):
    return re.sub(r"[!?.,;:]", " ", str(text))

def neutral_footer(text):
    return str(text) + (
        "\n\nThis message is intended for its recipient."
    )

transforms = {
    "original": lambda x: str(x),
    "uppercase": uppercase,
    "extra_spaces": extra_spaces,
    "punctuation_removed": punctuation_removed,
    "neutral_footer": neutral_footer
}

original_predictions = model.predict(df["text"])

rows = []

print("\n=== CYB-5 EXTERNAL ROBUSTNESS ===")
print("External phishing samples:", len(df))

for name, transform in transforms.items():
    modified_text = df["text"].map(transform)

    predictions = model.predict(modified_text)

    detected = int((predictions == 1).sum())
    missed = int((predictions == 0).sum())

    flipped = int(
        (predictions != original_predictions).sum()
    )

    # Evasion among phishing samples originally detected.
    originally_detected = original_predictions == 1
    newly_missed = int(
        (originally_detected & (predictions == 0)).sum()
    )

    recovered = int(
        ((original_predictions == 0) & (predictions == 1)).sum()
    )

    recall = recall_score(
        df["label"], predictions, zero_division=0
    )

    rows.append({
        "transformation": name,
        "detected": detected,
        "missed": missed,
        "recall": float(recall),
        "prediction_flips": flipped,
        "newly_missed": newly_missed,
        "recovered": recovered
    })

summary = pd.DataFrame(rows)

print("\n=== RESULTS ===")
print(summary.to_string(index=False))

summary.to_csv(
    METRICS / "external_robustness.csv",
    index=False
)

with open(
    METRICS / "external_robustness.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(rows, f, indent=4)

print("\nExternal robustness analysis completed.")
