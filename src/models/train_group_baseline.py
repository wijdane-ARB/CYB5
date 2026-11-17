
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

MODELS.mkdir(exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 GROUP-AWARE BASELINE ===")

ceas = pd.read_csv(DATA / "ceas_clean.csv")

def load_split(name):
    ids = pd.read_csv(
        METRICS / f"group_{name}_ids.csv"
    )

    if ids["text_hash"].duplicated().any():
        raise ValueError(f"Duplicate hashes in {name}")

    split = ceas.merge(
        ids[["text_hash", "group"]],
        on="text_hash",
        how="inner",
        validate="one_to_one"
    )

    if len(split) != len(ids):
        raise ValueError(f"{name}: reconstruction mismatch")

    return split

train = load_split("train")
validation = load_split("validation")
test = load_split("test")

# Verify group separation.
for a, b in [
    (train, validation),
    (train, test),
    (validation, test)
]:
    assert set(a["group"]).isdisjoint(set(b["group"]))

print("\nTraining:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

# Same model configuration as original baseline.
model = Pipeline([
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

print("\nTraining group-aware model...")
model.fit(train["text"], train["label"])
print("Training completed!")

def evaluate(name, dataset):
    y_true = dataset["label"].astype(int)
    y_pred = model.predict(dataset["text"])

    print(f"\n=== {name.upper()} RESULTS ===")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        zero_division=0
    ))

    cm = confusion_matrix(
        y_true, y_pred,
        labels=[0, 1]
    )

    print("Confusion matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    return {
        "samples": int(len(dataset)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "false_positive_rate": float(
            fp / (fp + tn)
        ) if fp + tn else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

validation_result = evaluate("group_validation", validation)
test_result = evaluate("group_test", test)

# Exploratory external evaluation.
nazario = pd.read_csv(
    DATA / "nazario_external.csv"
).dropna(subset=["text", "label"])

if set(nazario["label"].astype(int).unique()) != {1}:
    raise ValueError("Nazario must be positive-only")

external_pred = model.predict(nazario["text"])

external_recall = float(
    recall_score(
        nazario["label"],
        external_pred,
        zero_division=0
    )
)

print("\n=== NAZARIO EXTERNAL RESULTS ===")
print("Samples:", len(nazario))
print("Phishing recall:", round(external_recall, 4))

results = {
    "experiment": "group_aware_baseline",
    "model": "TF-IDF + Logistic Regression",
    "training_size": len(train),
    "validation": validation_result,
    "test": test_result,
    "nazario_exploratory_recall": external_recall,
    "notes": (
        "Approximate MinHash LSH groups; "
        "Nazario used for exploratory evaluation only."
    )
}

output = METRICS / "group_baseline_results.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

joblib.dump(
    model,
    MODELS / "group_tfidf_lr.joblib"
)

print("\n=== GROUP BASELINE COMPLETED ===")
print("Results:", output)
print("Model saved successfully.")
