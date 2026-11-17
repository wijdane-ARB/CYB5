
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

ceas = pd.read_csv(DATA / "ceas_clean.csv")

# Load the NEW group-aware partitions
def get_split(name):
    ids = pd.read_csv(
        METRICS / f"group_{name}_ids.csv"
    )

    result = ceas.merge(
        ids[["text_hash"]].drop_duplicates(),
        on="text_hash",
        how="inner",
        validate="one_to_one"
    )

    if len(result) != len(ids):
        raise ValueError(
            f"{name}: group split reconstruction mismatch"
        )

    return result

train = get_split("train")
test = get_split("test")

print("=== CYB-5 GROUP SIMILARITY AUDIT ===")
print("Training samples:", len(train))
print("Testing samples:", len(test))

# Character-level TF-IDF for similarity detection
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=50000,
    dtype=np.float32
)

print("\nVectorizing emails...")

X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

# Find the most similar training message for each test email
print("\nSearching for nearest training emails...")

neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine",
    algorithm="brute",
    n_jobs=2
)

neighbors.fit(X_train)

distances, indices = neighbors.kneighbors(X_test)

similarities = np.clip(
    1 - distances.ravel(), 0, 1
)

print("\n=== SIMILARITY DISTRIBUTION ===")
print(pd.Series(similarities).describe().round(4))

print("\n=== SIMILARITY THRESHOLDS ===")

for threshold in [0.80, 0.90, 0.95, 0.99]:
    count = int((similarities >= threshold).sum())
    percentage = 100 * count / len(test)

    print(
        f"Similarity >= {threshold}: "
        f"{count}/{len(test)} ({percentage:.2f}%)"
    )

# Save detailed audit results
results = pd.DataFrame({
    "test_hash": test["text_hash"].values,
    "test_label": test["label"].values,
    "nearest_train_hash": train.iloc[
        indices.ravel()
    ]["text_hash"].values,
    "similarity": similarities
})

output = METRICS / "group_near_duplicate_audit.csv"

results.to_csv(output, index=False)

print("\n=== AUDIT COMPLETED ===")
print("Results saved to:", output)
