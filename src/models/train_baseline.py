
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

for folder in [METRICS, FIGURES, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42

print("=== CYB-5 BASELINE TRAINING ===")

# 1. Load CEAS dataset
df = pd.read_csv(DATA / "ceas_clean.csv")
df = df.dropna(subset=["text", "label"])

# Deduplicate normalized text before splitting
df["normalized"] = (
    df["text"].astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df = df.drop_duplicates(subset="normalized")
df = df.drop(columns=["normalized"])

print("\nDataset size:", len(df))
print("Label distribution:")
print(df["label"].value_counts())

# 2. Stratified 70/15/15 split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"]
)

print("\nTraining:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# 3. Define CPU-friendly pipeline
pipeline = Pipeline([
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
        random_state=SEED
    ))
])

# 4. Train
print("\nTraining TF-IDF + Logistic Regression...")
pipeline.fit(train_df["text"], train_df["label"])
print("Training completed!")

# 5. Evaluation
def evaluate(name, dataset):
    y_true = dataset["label"]
    y_pred = pipeline.predict(dataset["text"])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        ))
    }

    print(f"\n=== {name.upper()} RESULTS ===")
    print(classification_report(
        y_true, y_pred, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Spam/Phishing"]
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"CYB-5 - {name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{name}_confusion_matrix.png",
        dpi=200
    )
    plt.close()

    return metrics

validation_metrics = evaluate("validation", val_df)
test_metrics = evaluate("test", test_df)

# 6. External phishing-only evaluation
nazario = pd.read_csv(DATA / "nazario_external.csv")
nazario = nazario.dropna(subset=["text", "label"])

external_predictions = pipeline.predict(nazario["text"])
external_recall = float(
    recall_score(
        nazario["label"],
        external_predictions,
        zero_division=0
    )
)

print("\n=== EXTERNAL NAZARIO EVALUATION ===")
print("Phishing samples:", len(nazario))
print("Phishing recall:", round(external_recall, 4))
print("Note: recall only; no legitimate samples.")

# 7. Save results
results = {
    "model": "TF-IDF + Logistic Regression",
    "training_corpus": "CEAS_08",
    "seed": SEED,
    "label_1": "spam_or_phishing",
    "train_size": len(train_df),
    "validation_size": len(val_df),
    "test_size": len(test_df),
    "validation": validation_metrics,
    "test": test_metrics,
    "nazario_external_phishing_recall": external_recall,
    "limitation": (
        "CEAS positive class includes generic spam. "
        "Nazario contains only positive samples."
    )
}

with open(
    METRICS / "baseline_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=4)

joblib.dump(pipeline, MODELS / "baseline_tfidf_lr.joblib")

# Save split identifiers, not raw sensitive email text
for name, split in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:
    split[["text_hash", "label"]].to_csv(
        METRICS / f"{name}_split_ids.csv",
        index=False
    )

print("\n=== CYB-5 BASELINE COMPLETED ===")
print("Model saved to:", MODELS)
print("Results saved to:", METRICS)
print("Figures saved to:", FIGURES)
