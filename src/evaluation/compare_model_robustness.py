
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_benchmark" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

METRICS.mkdir(parents=True, exist_ok=True)

test = pd.read_parquet(DATA / "meajor_test.parquet")

# Fixed stratified sample, identical for both models.
sample = (
    test.groupby("label", group_keys=False)
    .sample(n=500, random_state=42)
    .sort_values("text_hash")
    .reset_index(drop=True)
)

footer = "\n\nThis message is intended for its recipient."

models = {
    "Logistic Regression": (
        "meajor_tfidf_lr.joblib", 0.69
    ),
    "ComplementNB": (
        "meajor_complement_nb.joblib", 0.64
    )
}

labels = sample["label"].astype(int).to_numpy()
clean_text = sample["text"].astype(str)
modified_text = clean_text + footer

rows = []

print("=== CYB-5 MODEL ROBUSTNESS COMPARISON ===")
print("Samples:", len(sample))

for name, (filename, threshold) in models.items():
    model = joblib.load(MODELS / filename)

    clean_pred = (
        model.predict_proba(clean_text)[:, 1] >= threshold
    ).astype(int)

    modified_pred = (
        model.predict_proba(modified_text)[:, 1] >= threshold
    ).astype(int)

    for condition, pred in [
        ("clean", clean_pred),
        ("neutral_footer", modified_pred)
    ]:
        tn, fp, fn, tp = confusion_matrix(
            labels, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "model": name,
            "condition": condition,
            "accuracy": accuracy_score(labels, pred),
            "recall": recall_score(labels, pred),
            "fpr": fp / (fp + tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "prediction_flips_from_clean": int(
                (pred != clean_pred).sum()
            ),
            "newly_missed_positives": int((
                (labels == 1) &
                (clean_pred == 1) &
                (pred == 0)
            ).sum())
        })

results = pd.DataFrame(rows)

print("\n=== RESULTS ===")
print(results.round(4).to_string(index=False))

results.to_csv(
    METRICS / "model_robustness_comparison.csv",
    index=False
)

print("\n=== EXPERIMENT 21 COMPLETED ===")
