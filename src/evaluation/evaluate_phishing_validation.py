
from pathlib import Path
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# CYB-5 - EXTERNAL PHISHING EVALUATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "phishing_validation"
TREC = (
    ROOT / "data" / "phishing_benchmark"
    / "processed" / "meajor_train.parquet"
)
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

METRICS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 PHISHING VALIDATION ===")

# 1. Load emails and metadata
emails = pd.read_csv(
    DATA / "Phishing_validation_emails_v3.csv"
)

metadata = pd.read_csv(
    DATA / "phishing_v3_metadata.csv"
)

if len(emails) != len(metadata):
    raise ValueError("Dataset row counts differ")

if metadata["record_id"].tolist() != list(
    range(1, len(emails) + 1)
):
    raise ValueError("Metadata row alignment failed")

labels_email = (
    emails["Email Type"].astype(str).str.strip()
)
labels_meta = (
    metadata["email_type"].astype(str).str.strip()
)

if not labels_email.equals(labels_meta):
    raise ValueError("Email and metadata labels differ")

# 2. Explicitly map class labels
label_mapping = {
    "Safe Email": 0,
    "Phishing Email": 1
}

if not set(labels_email.unique()).issubset(
    label_mapping
):
    raise ValueError(
        f"Unexpected labels: {labels_email.unique()}"
    )

df = pd.DataFrame({
    "text": emails["Email Text"],
    "label": labels_email.map(label_mapping),
    "source": metadata["source_category"],
    "template_id": metadata["template_id"]
})

if df[["text", "label", "source"]].isna().any().any():
    raise ValueError("Missing critical data")

df["text"] = df["text"].astype(str)

print("\nTotal emails:", len(df))
print("Label distribution:")
print(df["label"].value_counts().sort_index())

print("\nSource distribution:")
print(df["source"].value_counts())

# 3. Check exact normalized overlap with TREC training.
def fingerprint(text):
    normalized = re.sub(
        r"\s+", " ", str(text).lower()
    ).strip()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

train = pd.read_parquet(TREC)

training_hashes = set(
    train["text"].map(fingerprint)
)
external_hashes = df["text"].map(fingerprint)

overlap = external_hashes.isin(training_hashes)

print("\nExact normalized overlap with training:",
      int(overlap.sum()))

if overlap.any():
    raise ValueError(
        "External dataset overlaps training data. "
        "Review before evaluation."
    )

# 4. Load frozen models
models = {
    "Logistic Regression": (
        "meajor_tfidf_lr.joblib", 0.69
    ),
    "ComplementNB": (
        "meajor_complement_nb.joblib", 0.64
    )
}

# 5. Compute metrics
def evaluate(y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "fpr": float(fp / (fp + tn))
        if fp + tn else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

rows = []

# 6. Evaluate each model, overall and by source
for model_name, (filename, threshold) in models.items():

    model = joblib.load(MODELS / filename)

    probabilities = model.predict_proba(
        df["text"]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    for source_name, mask in [
        ("all", np.ones(len(df), dtype=bool)),
        ("generated_v3",
         df["source"].eq("generated_v3").to_numpy()),
        ("original_v1",
         df["source"].eq("original_v1").to_numpy())
    ]:
        if mask.sum() == 0:
            continue

        y_true = df.loc[mask, "label"].to_numpy()
        y_pred = predictions[mask]

        result = evaluate(y_true, y_pred)

        rows.append({
            "model": model_name,
            "source": source_name,
            "threshold": threshold,
            **result
        })

# 7. Display and save results
summary = pd.DataFrame(rows)

print("\n=== EXTERNAL EVALUATION RESULTS ===")
print(summary.round(4).to_string(index=False))

csv_path = METRICS / "phishing_validation_results.csv"
json_path = METRICS / "phishing_validation_results.json"

summary.to_csv(csv_path, index=False)

json_path.write_text(
    json.dumps(
        rows,
        indent=2
    ),
    encoding="utf-8"
)

print("\n=== EXPERIMENT 23 COMPLETED ===")
print("CSV:", csv_path)
print("JSON:", json_path)
