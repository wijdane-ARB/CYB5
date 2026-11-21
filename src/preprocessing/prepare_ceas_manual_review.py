
from pathlib import Path
import pandas as pd

# ==========================================================
# CYB-5 — EXPERIMENT 37A
# CEAS BLINDED MANUAL LABEL-QUALITY AUDIT
#
# 40 original label-0 messages
# 40 original label-1 messages
#
# Source: existing TRAIN + VALIDATION partitions only.
# TEST is excluded.
#
# The original label key is kept separate from reviewers.
# Full text is stored as local .txt files.
# No existing dataset or model is modified.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

CEAS_FILE = ROOT / "data" / "processed" / "ceas_clean.csv"

SPLITS_DIR = ROOT / "results" / "metrics"

TRAIN_IDS = SPLITS_DIR / "group_train_ids.csv"
VAL_IDS = SPLITS_DIR / "group_validation_ids.csv"
TEST_IDS = SPLITS_DIR / "group_test_ids.csv"

AUDIT_DIR = ROOT / "data" / "audit" / "ceas_stageA"

BLIND_DIR = AUDIT_DIR / "blind"
MESSAGES_DIR = BLIND_DIR / "messages"
PRIVATE_DIR = AUDIT_DIR / "private"

SEED = 42
SAMPLES_PER_LABEL = 40

print("=" * 60)
print("CYB-5 — EXPERIMENT 37A")
print("BLINDED CEAS LABEL-QUALITY AUDIT")
print("=" * 60)

# ----------------------------------------------------------
# 1. Verify the required input files
# ----------------------------------------------------------

required_files = [
    CEAS_FILE,
    TRAIN_IDS,
    VAL_IDS,
    TEST_IDS,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

# ----------------------------------------------------------
# 2. Load the cleaned CEAS dataset
# ----------------------------------------------------------

df = pd.read_csv(CEAS_FILE)

required_columns = {
    "text",
    "label",
    "text_hash",
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing CEAS columns: "
        f"{required_columns - set(df.columns)}"
    )

if df["text_hash"].duplicated().any():
    raise ValueError(
        "Duplicate text hashes in cleaned CEAS."
    )

print("\n=== CEAS DATASET ===")
print("Total emails:", len(df))

# ----------------------------------------------------------
# 3. Load the existing grouped split identifiers
# ----------------------------------------------------------

def load_hashes(path):
    split = pd.read_csv(path, dtype={"text_hash": str})

    if "text_hash" not in split.columns:
        raise ValueError(
            f"text_hash missing from {path.name}"
        )

    hashes = set(split["text_hash"].dropna())

    if len(hashes) != len(split):
        raise ValueError(
            f"Repeated or missing hashes in {path.name}"
        )

    return hashes


train_hashes = load_hashes(TRAIN_IDS)
val_hashes = load_hashes(VAL_IDS)
test_hashes = load_hashes(TEST_IDS)

if (
    train_hashes & val_hashes
    or train_hashes & test_hashes
    or val_hashes & test_hashes
):
    raise RuntimeError(
        "Overlap between split identifiers detected."
    )

all_hashes = set(df["text_hash"])

if (train_hashes | val_hashes | test_hashes) != all_hashes:
    raise RuntimeError(
        "Grouped split identifiers do not match "
        "the cleaned CEAS dataset."
    )

print("\n=== GROUP SPLIT VERIFICATION ===")
print("Train:", len(train_hashes))
print("Validation:", len(val_hashes))
print("Test:", len(test_hashes))
print("Hash overlap between splits: 0")

# ----------------------------------------------------------
# 4. Restrict manual review to train + validation
# ----------------------------------------------------------

allowed_hashes = train_hashes | val_hashes

eligible = df[
    df["text_hash"].isin(allowed_hashes)
].copy()

eligible["partition"] = eligible["text_hash"].map(
    lambda h: (
        "train" if h in train_hashes else "validation"
    )
)

print("\n=== ELIGIBLE AUDIT POPULATION ===")
print("Total:", len(eligible))
print("Label distribution:")
print(
    eligible["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

# ----------------------------------------------------------
# 5. Random sample per original label
# ----------------------------------------------------------

selected_parts = []

for label in [0, 1]:

    group = eligible[
        eligible["label"] == label
    ]

    if len(group) < SAMPLES_PER_LABEL:
        raise ValueError(
            f"Not enough examples for label {label}"
        )

    selected_parts.append(
        group.sample(
            n=SAMPLES_PER_LABEL,
            random_state=SEED + label
        )
    )

sample = pd.concat(
    selected_parts,
    ignore_index=True
)

# Shuffle so the reviewer cannot infer label
# from the order of the messages.

sample = sample.sample(
    frac=1,
    random_state=SEED + 100
).reset_index(drop=True)

sample["review_id"] = [
    f"R{i:03d}"
    for i in range(1, len(sample) + 1)
]

# Never include original label in the reviewer sheet.

if sample["text_hash"].isin(test_hashes).any():
    raise RuntimeError(
        "TEST data entered the manual audit!"
    )

# ----------------------------------------------------------
# 6. Avoid silently overwriting previous annotations
# ----------------------------------------------------------

if AUDIT_DIR.exists():
    existing = list(AUDIT_DIR.rglob("*"))

    if any(p.is_file() for p in existing):
        raise FileExistsError(
            "Audit files already exist. "
            "Do not overwrite previous reviewer work. "
            f"Check: {AUDIT_DIR}"
        )

MESSAGES_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 7. Save the PRIVATE label key
# ----------------------------------------------------------

sample["has_visible_url"] = (
    sample["text"].astype(str)
    .str.contains(
        r"https?://|www\.",
        case=False,
        regex=True
    )
)

private_key = sample[
    [
        "review_id",
        "text_hash",
        "label",
        "partition",
        "has_visible_url",
    ]
].copy()

private_key.to_csv(
    PRIVATE_DIR / "ceas_review_key.csv",
    index=False
)

# ----------------------------------------------------------
# 8. Save COMPLETE texts for blinded reviewers
# ----------------------------------------------------------

for row in sample.itertuples(index=False):

    output_file = (
        MESSAGES_DIR / f"{row.review_id}.txt"
    )

    output_file.write_text(
        str(row.text),
        encoding="utf-8"
    )

# ----------------------------------------------------------
# 9. Create independent annotation sheets
# ----------------------------------------------------------

annotations = pd.DataFrame({
    "review_id": sample["review_id"],
    "judgment": "",
    "evidence": "",
    "notes": "",
})

annotations.to_csv(
    BLIND_DIR / "reviewer_1.csv",
    index=False,
    encoding="utf-8-sig"
)

annotations.to_csv(
    BLIND_DIR / "reviewer_2.csv",
    index=False,
    encoding="utf-8-sig"
)

# ----------------------------------------------------------
# 10. Final verification
# ----------------------------------------------------------

print("\n=== SAMPLE CREATED ===")
print("Total review messages:", len(sample))
print("Label 0:", int((sample["label"] == 0).sum()))
print("Label 1:", int((sample["label"] == 1).sum()))
print(
    "Test messages included:",
    int(sample["text_hash"].isin(test_hashes).sum())
)

print("\n=== OUTPUT LOCATIONS ===")
print("Reviewer messages:", MESSAGES_DIR)
print("Reviewer sheet 1:", BLIND_DIR / "reviewer_1.csv")
print("Reviewer sheet 2:", BLIND_DIR / "reviewer_2.csv")
print("Private label key:", PRIVATE_DIR / "ceas_review_key.csv")

print("\n=== EXPERIMENT 37A PREPARATION COMPLETED ===")
print("Do not publish the audit files.")
print("Do not inspect the private key while annotating.")