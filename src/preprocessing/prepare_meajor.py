
from pathlib import Path
import hashlib
import json
import re

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "phishing_benchmark"
    / "meajor_cleaned_preprocessed.parquet.gzip"
)

OUT = ROOT / "data" / "phishing_benchmark" / "processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 MEAJOR PREPARATION ===")

parquet = pq.ParquetFile(INPUT)

rows = []
excluded = 0

for batch in parquet.iter_batches(
    batch_size=5000,
    columns=["subject", "body", "source", "label"]
):
    for record in batch.to_pylist():

        source = record["source"]
        label = record["label"]
        body = record["body"]

        if source is None or label is None or body is None:
            excluded += 1
            continue

        if source not in {"trec5", "trec6", "trec7"}:
            excluded += 1
            continue

        label = int(label)

        if label not in (0, 1):
            excluded += 1
            continue

        subject = record["subject"] or ""
        text = (subject + "\n" + body).strip()

        if len(text) < 20:
            excluded += 1
            continue

        normalized = re.sub(
            r"\s+", " ", text.lower()
        ).strip()

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        rows.append({
            "text": text,
            "label": label,
            "source": source,
            "text_hash": digest
        })

df = pd.DataFrame(rows)

print("Valid rows before deduplication:", len(df))
print("Excluded rows:", excluded)

# Detect contradictory labels for identical text.
conflicts = (
    df.groupby("text_hash")["label"]
    .nunique()
)

bad_hashes = set(conflicts[conflicts > 1].index)

df = df.loc[
    ~df["text_hash"].isin(bad_hashes)
].copy()

print("Conflicting hashes removed:", len(bad_hashes))

# Prioritize retention of test samples, then validation.
priority = {
    "trec7": 0,
    "trec6": 1,
    "trec5": 2
}

df["priority"] = df["source"].map(priority)
df = df.sort_values("priority", kind="stable")

before = len(df)
df = df.drop_duplicates(subset="text_hash")

print("Duplicate rows removed:", before - len(df))

splits = {
    "train": "trec5",
    "validation": "trec6",
    "test": "trec7"
}

summary = {
    "excluded_rows": excluded,
    "conflicting_hashes": len(bad_hashes),
    "duplicate_rows_removed": before - len(df),
    "splits": {}
}

for name, source in splits.items():

    subset = df.loc[
        df["source"] == source,
        ["text", "label", "source", "text_hash"]
    ].copy()

    subset.to_parquet(
        OUT / f"meajor_{name}.parquet",
        index=False
    )

    counts = subset["label"].value_counts()

    summary["splits"][name] = {
        "source": source,
        "samples": len(subset),
        "label_0": int(counts.get(0, 0)),
        "label_1": int(counts.get(1, 0))
    }

    print(f"\n=== {name.upper()} ===")
    print("Source:", source)
    print("Samples:", len(subset))
    print(subset["label"].value_counts())

# Verify that no exact normalized text hash crosses splits.
assert df["text_hash"].is_unique

output = REPORTS / "meajor_preparation.json"

output.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== PREPARATION COMPLETED ===")
print("Processed datasets:", OUT)
print("Summary:", output)
