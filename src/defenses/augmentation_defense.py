
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

METRICS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

ceas = pd.read_csv(DATA / "ceas_clean.csv")
nazario = pd.read_csv(DATA / "nazario_external.csv")

def get_split(name):
    ids = pd.read_csv(
        METRICS / f"{name}_split_ids.csv"
    )

    result = ceas.merge(
        ids[["text_hash"]].drop_duplicates(),
        on="text_hash",
        how="inner",
        validate="one_to_one"
    )

    if len(result) != len(ids):
        raise ValueError(f"{name} split mismatch")

    return result

train = get_split("train")
test = get_split("test")

original = joblib.load(
    MODELS / "baseline_tfidf_lr.joblib"
)

# Training-only augmentation templates.
training_footers = [
    "\n\nPlease keep this correspondence for your records.",
    "\n\nThis notification was generated for information purposes.",
    "\n\nFor administrative reference only."
]

rng = np.random.default_rng(42)

# Add an augmented copy of each training email.
augmented = train.copy()

chosen = rng.integers(
    0, len(training_footers), size=len(train)
)

augmented["text"] = [
    text + training_footers[i]
    for text, i in zip(train["text"], chosen)
]

combined = pd.concat(
    [train, augmented],
    ignore_index=True
)

print("=== CYB-5 AUGMENTATION DEFENSE ===")
print("Original training samples:", len(train))
print("Augmented training rows:", len(combined))

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ))
    ])

defended = build_model()

print("\nTraining defended model...")
defended.fit(combined["text"], combined["label"])
print("Training completed!")

# Held-out footer, not used during augmentation.
evaluation_footer = (
    "\n\nThis message is intended for its recipient."
)

def evaluate(model, df, name):
    clean = df["text"].astype(str)
    modified = clean + evaluation_footer
    labels = df["label"].astype(int).to_numpy()

    clean_pred = model.predict(clean)
    modified_pred = model.predict(modified)

    originally_detected = (
        (labels == 1) & (clean_pred == 1)
    )

    newly_missed = int((
        originally_detected & (modified_pred == 0)
    ).sum())

    detected_count = int(originally_detected.sum())

    return {
        "model": name,
        "clean_accuracy": float(
            accuracy_score(labels, clean_pred)
        ),
        "modified_accuracy": float(
            accuracy_score(labels, modified_pred)
        ),
        "clean_recall": float(
            recall_score(labels, clean_pred)
        ),
        "modified_recall": float(
            recall_score(labels, modified_pred)
        ),
        "newly_missed": newly_missed,
        "evasion_among_detected": (
            newly_missed / detected_count
            if detected_count else None
        )
    }

results = []

for model_name, model in [
    ("Original", original),
    ("Augmented", defended)
]:
    for dataset_name, dataset in [
        ("CEAS_test", test),
        ("Nazario_exploratory", nazario)
    ]:
        row = evaluate(model, dataset, model_name)
        row["dataset"] = dataset_name
        results.append(row)

summary = pd.DataFrame(results)

print("\n=== DEFENSE COMPARISON ===")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    METRICS / "augmentation_defense.csv",
    index=False
)

joblib.dump(
    defended,
    MODELS / "augmented_tfidf_lr.joblib"
)

print("\nDefense experiment completed.")
