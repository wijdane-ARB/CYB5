
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"

METRICS.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_PERM = 64
LSH_THRESHOLD = 0.80

df = pd.read_csv(DATA / "ceas_clean.csv")
df = df.dropna(subset=["text", "label", "text_hash"])
df = df.drop_duplicates(subset="text_hash").reset_index(drop=True)

print("=== CYB-5 GROUP-AWARE SPLIT ===")
print("Total emails:", len(df))

def normalize(text):
    text = re.sub(r"\s+", " ", str(text).lower()).strip()
    return text

def shingles(text, k=5):
    words = normalize(text).split()

    if len(words) < k:
        return {" ".join(words)}

    return {
        " ".join(words[i:i + k])
        for i in range(len(words) - k + 1)
    }

# Disjoint-set union for grouping related messages.
parent = list(range(len(df)))
rank = [0] * len(df)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        ra, rb = rb, ra

    parent[rb] = ra

    if rank[ra] == rank[rb]:
        rank[ra] += 1

lsh = MinHashLSH(
    threshold=LSH_THRESHOLD,
    num_perm=NUM_PERM
)

signatures = []

print("\nGenerating MinHash signatures...")

for i, text in enumerate(df["text"]):
    mh = MinHash(
        num_perm=NUM_PERM,
        seed=SEED
    )

    for shingle in shingles(text):
        mh.update(shingle.encode("utf-8"))

    signatures.append(mh)

    matches = lsh.query(mh)

    for match in matches:
        union(i, int(match))

    lsh.insert(str(i), mh)

    if (i + 1) % 5000 == 0:
        print(f"Processed {i + 1}/{len(df)}")

# Assign group identifiers.
df["group"] = [
    find(i) for i in range(len(df))
]

print("\nTotal similarity groups:", df["group"].nunique())

# Group-aware 70/15/15 split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=SEED
)

train_idx, temp_idx = next(
    splitter.split(
        df,
        groups=df["group"]
    )
)

train = df.iloc[train_idx]
temp = df.iloc[temp_idx]

second_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEED
)

val_idx, test_idx = next(
    second_splitter.split(
        temp,
        groups=temp["group"]
    )
)

validation = temp.iloc[val_idx]
test = temp.iloc[test_idx]

# Verify disjoint groups.
assert set(train["group"]).isdisjoint(validation["group"])
assert set(train["group"]).isdisjoint(test["group"])
assert set(validation["group"]).isdisjoint(test["group"])

for name, split in [
    ("group_train", train),
    ("group_validation", validation),
    ("group_test", test)
]:
    split[["text_hash", "label", "group"]].to_csv(
        METRICS / f"{name}_ids.csv",
        index=False
    )

    print(f"\n{name}: {len(split)} rows")
    print(split["label"].value_counts())

summary = {
    "total": len(df),
    "groups": int(df["group"].nunique()),
    "train": len(train),
    "validation": len(validation),
    "test": len(test),
    "threshold": LSH_THRESHOLD,
    "num_perm": NUM_PERM
}

with open(
    METRICS / "group_split_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n=== GROUP SPLIT COMPLETED ===")
print(json.dumps(summary, indent=2))
