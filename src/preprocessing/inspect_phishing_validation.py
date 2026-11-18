
from pathlib import Path
import json
import pandas as pd

# ==================================================
# CYB-5 - PHISHING VALIDATION DATASET INSPECTION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "phishing_validation"
REPORTS = ROOT / "reports"

EMAILS_FILE = DATA / "Phishing_validation_emails_v3.csv"
METADATA_FILE = DATA / "phishing_v3_metadata.csv"

REPORTS.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 PHISHING VALIDATION INSPECTION ===")

# --------------------------------------------------
# 1. Verify files
# --------------------------------------------------

for file in [EMAILS_FILE, METADATA_FILE]:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found: {file}"
        )

print("\nBoth dataset files found successfully.")

# --------------------------------------------------
# 2. Load CSV files
# --------------------------------------------------

emails = pd.read_csv(EMAILS_FILE)
metadata = pd.read_csv(METADATA_FILE)

print("\n=== DATASET DIMENSIONS ===")
print("Emails:", emails.shape)
print("Metadata:", metadata.shape)

# --------------------------------------------------
# 3. Inspect columns
# --------------------------------------------------

print("\n=== EMAIL COLUMNS ===")
print(emails.columns.tolist())

print("\n=== METADATA COLUMNS ===")
print(metadata.columns.tolist())

# --------------------------------------------------
# 4. Validate required columns
# --------------------------------------------------

required_email_columns = {
    "Email Text",
    "Email Type"
}

required_metadata_columns = {
    "record_id",
    "email_type",
    "source_category",
    "template_id"
}

missing_email = (
    required_email_columns - set(emails.columns)
)

missing_metadata = (
    required_metadata_columns - set(metadata.columns)
)

if missing_email:
    raise ValueError(
        f"Missing email columns: {missing_email}"
    )

if missing_metadata:
    raise ValueError(
        f"Missing metadata columns: {missing_metadata}"
    )

print("\nRequired columns verified.")

# --------------------------------------------------
# 5. Verify dataset sizes
# --------------------------------------------------

if len(emails) != len(metadata):
    raise ValueError(
        "Email and metadata row counts differ."
    )

print("\n=== RECORD COUNTS ===")
print("Total email records:", len(emails))
print("Total metadata records:", len(metadata))

# --------------------------------------------------
# 6. Validate metadata alignment
# --------------------------------------------------

expected_ids = list(range(1, len(emails) + 1))

actual_ids = pd.to_numeric(
    metadata["record_id"],
    errors="raise"
).tolist()

if actual_ids != expected_ids:
    raise ValueError(
        "Metadata record IDs are not sequential."
    )

email_labels = (
    emails["Email Type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

metadata_labels = (
    metadata["email_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)

if not email_labels.reset_index(drop=True).equals(
    metadata_labels.reset_index(drop=True)
):
    raise ValueError(
        "Email labels do not match metadata labels."
    )

print("\nMetadata alignment verified.")

# --------------------------------------------------
# 7. Label distribution
# --------------------------------------------------

print("\n=== LABEL DISTRIBUTION ===")

label_counts = emails[
    "Email Type"
].value_counts(dropna=False)

print(label_counts)

# --------------------------------------------------
# 8. Source distribution
# --------------------------------------------------

print("\n=== SOURCE DISTRIBUTION ===")

source_counts = metadata[
    "source_category"
].value_counts(dropna=False)

print(source_counts)

# --------------------------------------------------
# 9. Template information
# --------------------------------------------------

print("\n=== TEMPLATE INFORMATION ===")

unique_templates = metadata[
    "template_id"
].nunique(dropna=True)

print("Unique templates:", unique_templates)

print("\nTemplate distribution:")
print(
    metadata["template_id"]
    .value_counts()
    .head(10)
)

# --------------------------------------------------
# 10. Data quality
# --------------------------------------------------

print("\n=== DATA QUALITY ===")

missing_text = emails[
    "Email Text"
].isna().sum()

empty_text = (
    emails["Email Text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

duplicate_text = emails[
    "Email Text"
].duplicated().sum()

missing_labels = emails[
    "Email Type"
].isna().sum()

print("Missing text:", missing_text)
print("Empty text:", empty_text)
print("Duplicate emails:", duplicate_text)
print("Missing labels:", missing_labels)

# --------------------------------------------------
# 11. Save inspection report
# --------------------------------------------------

summary = {
    "dataset": "Phishing Validation Emails v3",
    "email_records": int(len(emails)),
    "metadata_records": int(len(metadata)),
    "email_columns": emails.columns.tolist(),
    "metadata_columns": metadata.columns.tolist(),
    "label_distribution": {
        str(k): int(v)
        for k, v in label_counts.items()
    },
    "source_distribution": {
        str(k): int(v)
        for k, v in source_counts.items()
    },
    "unique_templates": int(unique_templates),
    "missing_text": int(missing_text),
    "empty_text": int(empty_text),
    "duplicate_emails": int(duplicate_text),
    "missing_labels": int(missing_labels),
    "metadata_alignment_verified": True
}

OUTPUT = (
    REPORTS / "phishing_validation_inspection.json"
)

OUTPUT.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8"
)

print("\n=== INSPECTION COMPLETED ===")
print("Report saved to:", OUTPUT)
