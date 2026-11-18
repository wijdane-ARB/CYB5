
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_validation"
OUT = DATA / "processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42

print("=== CYB-5 TEMPLATE-AWARE SPLIT ===")

emails = pd.read_csv(
    DATA / "Phishing_validation_emails_v3.csv"
)

meta = pd.read_csv(
    DATA / "phishing_v3_metadata.csv"
)

if len(emails) != len(meta):
    raise ValueError("Row count mismatch")

if not emails["Email Type"].reset_index(drop=True).equals(
    meta["email_type"].reset_index(drop=True)
):
    raise ValueError("Metadata label mismatch")

if meta["record_id"].tolist() != list(
    range(1, len(meta) + 1)
):
    raise ValueError("Record ID alignment mismatch")

label_map = {
    "Safe Email": 0,
    "Phishing Email": 1
}

df = pd.DataFrame({
    "record_id": meta["record_id"],
    "text": emails["Email Text"],
    "label": emails["Email Type"].map(label_map),
    "source": meta["source_category"],
    "template_id": meta["template_id"]
})

if df.isna().any().any():
    raise ValueError("Missing required values")

df["text"] = df["text"].astype(str)

def fingerprint(text):
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

df["text_hash"] = df["text"].map(fingerprint)

if df["text_hash"].duplicated().any():
    raise ValueError("Duplicate normalized emails found")

generated = df[
    df["source"] == "generated_v3"
].copy()

original = df[
    df["source"] == "original_v1"
].copy()

print("\nGenerated samples:", len(generated))
print("Original samples:", len(original))

# Split complete template families separately per class.
# This helps keep both labels in each partition.
def split_one_class(data):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=SEED
    )

    train_idx, temporary_idx = next(
        splitter.split(
            data,
            groups=data["template_id"]
        )
    )

    train = data.iloc[train_idx].copy()
    temporary = data.iloc[temporary_idx].copy()

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED
    )

    val_idx, test_idx = next(
        splitter2.split(
            temporary,
            groups=temporary["template_id"]
        )
    )

    validation = temporary.iloc[val_idx].copy()
    test = temporary.iloc[test_idx].copy()

    return train, validation, test

parts = {
    "train": [],
    "validation": [],
    "test": []
}

for label in [0, 1]:
    subset = generated[
        generated["label"] == label
    ].copy()

    train, validation, test = split_one_class(subset)

    parts["train"].append(train)
    parts["validation"].append(validation)
    parts["test"].append(test)

splits = {}

for name, pieces in parts.items():
    splits[name] = (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

splits["original_exploratory"] = original

# Verify that generated template families never
# cross train/validation/test boundaries.
for a, b in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]:
    assert set(
        splits[a]["template_id"]
    ).isdisjoint(
        splits[b]["template_id"]
    )

# Verify no normalized text crosses any partition.
names = list(splits)

for i, a in enumerate(names):
    for b in names[i + 1:]:
        assert set(
            splits[a]["text_hash"]
        ).isdisjoint(
            splits[b]["text_hash"]
        )

summary = {
    "method": "Template-aware group splitting",
    "random_seed": SEED,
    "note": (
        "Generated data are synthetic. "
        "Original subset is exploratory and "
        "was previously evaluated."
    ),
    "partitions": {}
}

for name, split in splits.items():
    path = OUT / f"{name}.parquet"

    split.to_parquet(path, index=False)

    counts = split["label"].value_counts()

    summary["partitions"][name] = {
        "samples": int(len(split)),
        "safe": int(counts.get(0, 0)),
        "phishing": int(counts.get(1, 0)),
        "templates": int(split["template_id"].nunique())
    }

    print(f"\n=== {name.upper()} ===")
    print("Samples:", len(split))
    print("Templates:", split["template_id"].nunique())
    print("Safe:", int(counts.get(0, 0)))
    print("Phishing:", int(counts.get(1, 0)))

output = REPORTS / "phishing_template_split.json"

output.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== TEMPLATE SPLIT COMPLETED ===")
print("Report:", output)
