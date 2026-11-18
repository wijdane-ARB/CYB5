
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external" / "spaphish"

def normalize(text):
    return " ".join(str(text).lower().split())

def hash_text(text):
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()

print("=== CYB-5 EXPERIMENT 31 ===")
print("LABEL DOCUMENTATION AND OVERLAP AUDIT")

# 1. Inspect schema references to label definitions.
schema_path = EXTERNAL / "dataset_schema.json"

if schema_path.is_file():
    schema = json.loads(
        schema_path.read_text(encoding="utf-8-sig")
    )

    print("\n=== SCHEMA: LABEL REFERENCES ===")

    def inspect(obj, path="root"):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}"
                if "label" in key.lower():
                    print(new_path, ":", str(value)[:1500])
                inspect(value, new_path)

        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                inspect(value, f"{path}[{index}]")

    inspect(schema)
else:
    print("\nSchema file not found.")

# 2. Load external emails.
csv_files = list(EXTERNAL.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("Expected exactly one SpaPhish CSV.")

external = pd.read_csv(
    csv_files[0],
    sep=None,
    engine="python"
)

external["text"] = (
    external["subject"].fillna("").astype(str)
    + "\n"
    + external["body"].fillna("").astype(str)
)

external_hashes = set(
    external["text"].map(hash_text)
)

print("\nExternal samples:", len(external))
print("Unique external hashes:", len(external_hashes))

# 3. Check overlap with synthetic training emails.
synthetic_train = (
    ROOT / "data" / "phishing_validation"
    / "processed" / "train.parquet"
)

if synthetic_train.is_file():
    train = pd.read_parquet(synthetic_train)

    train_hashes = set(train["text"].map(hash_text))

    overlap = external_hashes & train_hashes

    print("\n=== SYNTHETIC TRAIN OVERLAP ===")
    print("Training samples:", len(train))
    print("Exact normalized overlap:", len(overlap))
else:
    print("\nSynthetic training file missing.")

# 4. Check overlap with TREC-05 training.
trec_train = (
    ROOT / "data" / "phishing_benchmark"
    / "processed" / "meajor_train.parquet"
)

if trec_train.is_file():
    trec = pd.read_parquet(trec_train)

    print("\n=== TREC TRAIN OVERLAP ===")
    print("TREC samples:", len(trec))
    print("TREC columns:", trec.columns.tolist())

    text_column = next(
        (
            name for name in ["text", "email_text", "body"]
            if name in trec.columns
        ),
        None
    )

    if text_column is not None:
        trec_hashes = set(
            trec[text_column].fillna("").map(hash_text)
        )

        overlap = external_hashes & trec_hashes
        print("Exact normalized overlap:", len(overlap))
    else:
        print("Overlap not checked: identify text column.")
else:
    print("\nTREC training file missing.")

print("\n=== AUDIT COMPLETED ===")
