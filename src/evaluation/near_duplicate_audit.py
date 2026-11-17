
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

ceas = pd.read_csv(DATA / "ceas_clean.csv")

def get_split(name):
    ids = pd.read_csv(
        METRICS / f"{name}_split_ids.csv"
    )

    result = ceas.merge(
        ids[["text_hash"]].drop_duplicates(),
        on="text_hash",
        how="inner"
    )

    if len(result) != len(ids):
        raise ValueError(f"{name}: split reconstruction mismatch")

    return result

train = get_split("train")
test = get_split("test")

print("=== CYB-5 NEAR-DUPLICATE AUDIT ===")
print("Training samples:", len(train))
print("Testing samples:", len(test))

# Use character n-grams to detect similar email templates.
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    dtype=np.float32
)

X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

print("\nFinding nearest training message for each test email...")

neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute",
    n_jobs=2
)

neighbors.fit(X_train)

distances, indices = neighbors.kneighbors(X_test)
similarities = 1 - distances.ravel()

print("\nSimilarity distribution:")
print(pd.Series(similarities).describe().round(4))

for threshold in [0.80, 0.90, 0.95, 0.99]:
    count = int((similarities >= threshold).sum())
    print(
        f"Similarity >= {threshold}: "
        f"{count} / {len(test)} "
        f"({100 * count / len(test):.2f}%)"
    )

results = pd.DataFrame({
    "test_hash": test["text_hash"].values,
    "test_label": test["label"].values,
    "nearest_train_hash": train.iloc[
        indices.ravel()
    ]["text_hash"].values,
    "similarity": similarities
})

results.to_csv(
    METRICS / "near_duplicate_audit.csv",
    index=False
)

print("\nAudit saved successfully.")
