
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
    classification_report,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT / "data" / "phishing_benchmark" / "processed"
)

METRICS = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

METRICS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 MEAJOR CROSS-SOURCE BASELINE ===")

# 1. Load source-separated datasets.
train = pd.read_parquet(DATA / "meajor_train.parquet")
validation = pd.read_parquet(
    DATA / "meajor_validation.parquet"
)
test = pd.read_parquet(DATA / "meajor_test.parquet")

for name, df, expected_source in [
    ("Train", train, "trec5"),
    ("Validation", validation, "trec6"),
    ("Test", test, "trec7")
]:
    if not df["source"].eq(expected_source).all():
        raise ValueError(f"{name}: unexpected source")

    if df[["text", "label", "text_hash"]].isna().any().any():
        raise ValueError(f"{name}: missing values")

    print(f"\n{name}: {len(df)} emails")
    print(df["label"].value_counts().sort_index())

# 2. Check that no identical normalized texts
# are shared across partitions.
train_hashes = set(train["text_hash"])
val_hashes = set(validation["text_hash"])
test_hashes = set(test["text_hash"])

assert train_hashes.isdisjoint(val_hashes)
assert train_hashes.isdisjoint(test_hashes)
assert val_hashes.isdisjoint(test_hashes)

print("\nExact normalized overlap: 0")

# 3. Build same model family as previous baseline.
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

# 4. Train only on TREC-05.
print("\nTraining on TREC-05...")
model.fit(train["text"], train["label"].astype(int))
print("Training completed!")

# 5. Evaluate without adjusting the threshold.
def evaluate(name, df):
    X = df["text"]
    y_true = df["label"].astype(int)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    scores = {
        "samples": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else 0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    print(f"\n=== {name.upper()} RESULTS ===")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        zero_division=0
    ))

    print("Confusion matrix:")
    print(cm)

    print("\nDetailed metrics:")
    for key, value in scores.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return scores

train_result = evaluate("TREC-05 Training", train)
validation_result = evaluate(
    "TREC-06 Validation", validation
)
test_result = evaluate("TREC-07 Test", test)

# 6. Save results and model.
results = {
    "experiment": "MeAJOR cross-source baseline",
    "model": "TF-IDF + Logistic Regression",
    "train_source": "trec5",
    "validation_source": "trec6",
    "test_source": "trec7",
    "decision_threshold": 0.5,
    "train": train_result,
    "validation": validation_result,
    "test": test_result,
    "limitation": (
        "TREC positive labels may include generic spam; "
        "phishing-only validity is not established."
    )
}

output = METRICS / "meajor_baseline_results.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

joblib.dump(
    model,
    MODELS / "meajor_tfidf_lr.joblib"
)

print("\n=== MEAJOR BASELINE COMPLETED ===")
print("Results:", output)
print("Model saved successfully.")
