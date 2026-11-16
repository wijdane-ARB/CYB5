
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

def load_dataset(filename, source):
    df = pd.read_csv(RAW / filename, low_memory=False)
    required = {"body", "subject", "label"}

    if not required.issubset(df.columns):
        raise ValueError(f"{filename}: required columns missing")

    df = df.copy()
    df["source"] = source

    # Preserve the original label, without guessing its meaning.
    df["label"] = pd.to_numeric(df["label"], errors="raise")

    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{filename}: unexpected label values")

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # Use subject and body as the classifier input.
    df["text"] = (
        df["subject"].str.strip()
        + "\n"
        + df["body"].str.strip()
    ).str.strip()

    df = df.loc[df["text"].str.len() >= 20].copy()

    # Exact-content fingerprint for leakage checks.
    df["text_hash"] = df["text"].map(
        lambda x: hashlib.sha256(
            x.encode("utf-8")
        ).hexdigest()
    )

    return df[["source", "text", "label", "text_hash"]]

ceas = load_dataset("CEAS_08.csv", "CEAS_08")
nazario = load_dataset("Nazario.csv", "Nazario")

print("\n=== BEFORE DEDUPLICATION ===")
for name, df in [("CEAS_08", ceas), ("Nazario", nazario)]:
    print(f"\n{name}: {len(df)} rows")
    print(df["label"].value_counts().sort_index())

# Check conflicting labels for identical messages.
combined = pd.concat([ceas, nazario], ignore_index=True)

conflicts = (
    combined.groupby("text_hash")["label"]
    .nunique()
)
conflicting_hashes = set(conflicts[conflicts > 1].index)

print("\nConflicting-content hashes:", len(conflicting_hashes))

# Exclude contradictory examples before training.
combined = combined.loc[
    ~combined["text_hash"].isin(conflicting_hashes)
].copy()

# Keep CEAS as the development corpus.
ceas_clean = combined.loc[
    combined["source"] == "CEAS_08"
].drop_duplicates(subset="text_hash")

# Keep Nazario as a separate external corpus.
nazario_clean = combined.loc[
    combined["source"] == "Nazario"
].drop_duplicates(subset="text_hash")

# Remove Nazario messages that also occur in CEAS.
nazario_clean = nazario_clean.loc[
    ~nazario_clean["text_hash"].isin(ceas_clean["text_hash"])
]

ceas_clean.to_csv(OUT / "ceas_clean.csv", index=False)
nazario_clean.to_csv(OUT / "nazario_external.csv", index=False)

summary = {
    "ceas_rows": len(ceas_clean),
    "ceas_labels": {
        str(k): int(v)
        for k, v in ceas_clean["label"].value_counts().items()
    },
    "nazario_external_rows": len(nazario_clean),
    "conflicting_hashes": len(conflicting_hashes),
    "label_semantics_verified": False,
    "train_test_split_created": False
}

(REPORTS / "preprocessing_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== PREPROCESSING COMPLETE ===")
print(json.dumps(summary, indent=2))
print("\nFiles saved in:", OUT)
