
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

ceas = pd.read_csv(DATA / "ceas_clean.csv")

def load_split(name):
    ids = pd.read_csv(
        METRICS / f"{name}_split_ids.csv"
    )
    df = ceas.merge(
        ids[["text_hash"]].drop_duplicates(),
        on="text_hash",
        how="inner"
    )

    if len(df) != len(ids):
        raise ValueError(f"{name}: split size mismatch")

    return df

train = load_split("train")
test = load_split("test")

def extract_features(df):
    lengths = df["text"].astype(str).str.len().to_numpy()
    return np.log1p(lengths).reshape(-1, 1)

X_train = extract_features(train)
X_test = extract_features(test)

y_train = train["label"].astype(int)
y_test = test["label"].astype(int)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("\n=== LENGTH-ONLY BASELINE ===")
print("Train:", len(train))
print("Test:", len(test))

print("\nAccuracy:", round(
    accuracy_score(y_test, pred), 4
))
print("F1:", round(f1_score(y_test, pred), 4))
print("ROC-AUC:", round(
    roc_auc_score(y_test, proba), 4
))

print("\nClassification report:")
print(classification_report(y_test, pred))

result = pd.DataFrame([{
    "model": "Length-only Logistic Regression",
    "accuracy": accuracy_score(y_test, pred),
    "f1": f1_score(y_test, pred),
    "roc_auc": roc_auc_score(y_test, proba)
}])

result.to_csv(
    METRICS / "length_baseline.csv",
    index=False
)

print("\nSaved: results/metrics/length_baseline.csv")
