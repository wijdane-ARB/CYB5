
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

model = joblib.load(
    ROOT / "models" / "baseline_tfidf_lr.joblib"
)

ceas = pd.read_csv(DATA / "ceas_clean.csv")
test_ids = pd.read_csv(
    METRICS / "test_split_ids.csv"
)

test = ceas.merge(
    test_ids[["text_hash"]].drop_duplicates(),
    on="text_hash",
    how="inner",
    validate="one_to_one"
)

if len(test) != len(test_ids):
    raise ValueError("Test split reconstruction mismatch")

# Reproducible, stratified evaluation sample.
sample = (
    test.groupby("label", group_keys=False)
    .sample(n=250, random_state=42)
    .sort_values("text_hash")
    .reset_index(drop=True)
)

def uppercase(text):
    return str(text).upper()

def extra_spaces(text):
    return re.sub(r" +", "   ", str(text))

def punctuation(text):
    return re.sub(r"[!?.,;:]", " ", str(text))

def neutral_footer(text):
    return str(text) + (
        "\n\nThis message is intended for its recipient."
    )

transformations = {
    "original": lambda x: str(x),
    "uppercase": uppercase,
    "extra_spaces": extra_spaces,
    "punctuation_removed": punctuation,
    "neutral_footer": neutral_footer
}

y_true = sample["label"].astype(int).to_numpy()
original_predictions = model.predict(sample["text"])
original_correct = original_predictions == y_true

results = []

print("\n=== CYB-5 ROBUSTNESS EXPERIMENT ===")
print("Samples:", len(sample))
print("Original correct:", int(original_correct.sum()))

for name, transform in transformations.items():
    modified = sample["text"].map(transform)
    predictions = model.predict(modified)

    changed = predictions != original_predictions

    # Errors introduced among previously correct predictions.
    new_errors = original_correct & (predictions != y_true)

    row = {
        "transformation": name,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "positive_recall": float(
            recall_score(y_true, predictions)
        ),
        "prediction_flip_rate": float(changed.mean()),
        "new_errors": int(new_errors.sum()),
        "new_error_rate_among_originally_correct": float(
            new_errors.sum() / original_correct.sum()
        ) if original_correct.sum() else None
    }

    results.append(row)

summary = pd.DataFrame(results)

print("\n=== RESULTS ===")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    METRICS / "robustness_results.csv",
    index=False
)

with open(
    METRICS / "robustness_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=4)

print("\nRobustness evaluation completed.")
