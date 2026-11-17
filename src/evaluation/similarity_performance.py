
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

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

train_ids = pd.read_csv(
    METRICS / "train_split_ids.csv"
)

similarities = pd.read_csv(
    METRICS / "near_duplicate_audit.csv"
)

test = ceas.merge(
    test_ids[["text_hash"]].drop_duplicates(),
    on="text_hash",
    how="inner",
    validate="one_to_one"
)

train_labels = ceas[
    ["text_hash", "label"]
].merge(
    train_ids[["text_hash"]].drop_duplicates(),
    on="text_hash",
    validate="one_to_one"
).rename(columns={
    "text_hash": "nearest_train_hash",
    "label": "nearest_train_label"
})

if len(test) != len(test_ids):
    raise ValueError("Test reconstruction failed")

if similarities["test_hash"].duplicated().any():
    raise ValueError("Duplicate test hashes in audit")

test = test.merge(
    similarities[
        ["test_hash", "nearest_train_hash", "similarity"]
    ],
    left_on="text_hash",
    right_on="test_hash",
    validate="one_to_one"
)

test = test.merge(
    train_labels,
    on="nearest_train_hash",
    validate="many_to_one"
)

if len(test) != len(test_ids):
    raise ValueError("Similarity audit coverage mismatch")

test["prediction"] = model.predict(test["text"])

test["same_label"] = (
    test["label"] == test["nearest_train_label"]
)

bins = [-np.inf, 0.80, 0.90, 0.95, 0.99, np.inf]
names = [
    "< 0.80",
    "0.80 - 0.90",
    "0.90 - 0.95",
    "0.95 - 0.99",
    ">= 0.99"
]

test["similarity_group"] = pd.cut(
    test["similarity"],
    bins=bins,
    labels=names,
    right=False
)

results = []

for name, group in test.groupby(
    "similarity_group",
    observed=True
):
    results.append({
        "similarity_group": str(name),
        "samples": len(group),
        "accuracy": accuracy_score(
            group["label"], group["prediction"]
        ),
        "f1": f1_score(
            group["label"],
            group["prediction"],
            zero_division=0
        ),
        "nearest_label_agreement": group[
            "same_label"
        ].mean()
    })

summary = pd.DataFrame(results)

print("\n=== CYB-5 SIMILARITY VS PERFORMANCE ===")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    METRICS / "similarity_performance.csv",
    index=False
)

print("\n=== OVERALL TEST ACCURACY ===")
print(round(
    accuracy_score(test["label"], test["prediction"]),
    4
))

print("\nResults saved successfully.")