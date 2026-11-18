
from pathlib import Path
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "external" / "spaphish"

csv_files = list(DATA.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError(
        f"Expected one CSV. Found {len(csv_files)}."
    )

df = pd.read_csv(
    csv_files[0],
    sep=None,
    engine="python"
)

print("=== CYB-5 EXPERIMENT 31 ===")
print("SPAPHISH LABEL AND TEXT AUDIT")

print("\n=== LABEL DISTRIBUTION ===")
print(df["Label"].value_counts(dropna=False))

print("\n=== LABEL DATA TYPES ===")
print(df["Label"].map(type).value_counts())

print("\n=== TEXT AVAILABILITY ===")
for column in ["subject", "body"]:
    print(
        column,
        "missing:", df[column].isna().sum(),
        "empty:",
        df[column].fillna("").astype(str)
          .str.strip().eq("").sum()
    )

# Combine subject and body for evaluation.
df["text"] = (
    df["subject"].fillna("").astype(str)
    + "\n"
    + df["body"].fillna("").astype(str)
)

def normalize(text):
    return " ".join(text.lower().split())

df["text_hash"] = df["text"].map(
    lambda t: hashlib.sha256(
        normalize(t).encode("utf-8")
    ).hexdigest()
)

print("\n=== TEXT DUPLICATES ===")
print(
    "Exact normalized duplicates:",
    df["text_hash"].duplicated().sum()
)

print("\n=== TEXT LENGTH ===")
print(
    df["text"].str.len().describe().round(2)
)

print("\n=== SAMPLE IDENTIFIERS ===")
print(
    "Email IDs:",
    df["u\u00feeffhash"].head(5).tolist()
    if "u\u00feeffhash" in df.columns
    else "No identifier column detected"
)

print("\n=== AUDIT COMPLETED ===")
