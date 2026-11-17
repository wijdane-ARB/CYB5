
from pathlib import Path
import re
import json

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

MODELS.mkdir(exist_ok=True)

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
        raise ValueError(
            f"{name}: split reconstruction mismatch"
        )

    return result

train = get_split("train")
validation = get_split("validation")
test = get_split("test")

# Mask selected surface-level identifiers.
def normalize_email(text):
    text = str(text).lower()

    text = re.sub(
        r"https?://[^\s<>]+|www\.[^\s<>]+",
        " URLTOKEN ",
        text
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        " EMAILTOKEN ",
        text
    )

    text = re.sub(
        r"\b\d+\b",
        " NUMTOKEN ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

print("=== CYB-5 FEATURE ABLATION ===")

print("Normalizing training data...")
X_train = train["text"].map(normalize_email)

print("Normalizing validation data...")
X_val = validation["text"].map(normalize_email)

print("Normalizing test data...")
X_test = test["text"].map(normalize_email)

print("Normalizing external phishing data...")
X_external = nazario["text"].map(normalize_email)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ))
])

print("\nTraining normalized-text model...")
model.fit(X_train, train["label"])
print("Training completed!")

def evaluate(name, texts, labels):
    pred = model.predict(texts)

    scores = {
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
    }

    print(f"\n=== {name.upper()} ===")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")

    return {k: float(v) for k, v in scores.items()}

val_results = evaluate(
    "CEAS validation", X_val, validation["label"]
)

test_results = evaluate(
    "CEAS test", X_test, test["label"]
)

external_results = evaluate(
    "Nazario external", X_external, nazario["label"]
)

results = {
    "experiment": "normalized_text_ablation",
    "normalizations": [
        "URL masking",
        "Email address masking",
        "Number masking",
        "Lowercasing"
    ],
    "validation": val_results,
    "test": test_results,
    "nazario": external_results
}

with open(
    METRICS / "ablation_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=4)

joblib.dump(
    model,
    MODELS / "normalized_tfidf_lr.joblib"
)

print("\n=== ABLATION STUDY COMPLETED ===")
print("Results saved successfully.")
