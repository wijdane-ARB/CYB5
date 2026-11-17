
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

model = joblib.load(
    ROOT / "models" / "group_tfidf_lr.joblib"
)

ceas = pd.read_csv(DATA / "ceas_clean.csv")

ids = pd.read_csv(
    METRICS / "group_test_ids.csv"
)

similarities = pd.read_csv(
    METRICS / "group_near_duplicate_audit.csv"
)

test = ceas.merge(
    ids[["text_hash"]].drop_duplicates(),
    on="text_hash",
    how="inner",
    validate="one_to_one"
)

test = test.merge(
    similarities[["test_hash", "similarity"]],
    left_on="text_hash",
    right_on="test_hash",
    validate="one_to_one"
)

if len(test) != len(ids):
    raise ValueError("Test reconstruction failed")

print("=== CYB-5 LOW-SIMILARITY EVALUATION ===")

rows = []

for name, mask in [
    ("All group-test emails", test["similarity"] >= 0),
    ("Similarity < 0.80", test["similarity"] < 0.80),
    ("Similarity < 0.90", test["similarity"] < 0.90),
    ("Similarity < 0.95", test["similarity"] < 0.95)
]:
    subset = test.loc[mask]

    if subset.empty:
        continue

    pred = model.predict(subset["text"])
    labels = subset["label"].astype(int)

    rows.append({
        "subset": name,
        "samples": len(subset),
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(
            labels, pred, zero_division=0
        ),
        "recall": recall_score(
            labels, pred, zero_division=0
        ),
        "f1": f1_score(
            labels, pred, zero_division=0
        )
    })

summary = pd.DataFrame(rows)

print(summary.round(4).to_string(index=False))

summary.to_csv(
    METRICS / "group_low_similarity_results.csv",
    index=False
)

print("\nEvaluation completed.")
