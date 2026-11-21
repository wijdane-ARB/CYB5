
from pathlib import Path
from collections import Counter
from urllib.parse import urlsplit
import json
import re

import pandas as pd

# ============================================================
# CYB-5 — EXPERIMENT 36
# CEAS READINESS: LABELS, URL COVERAGE, ATTACK ELIGIBILITY
#
# This script does not train models or modify dataset splits.
# Heuristic phishing indicators are NOT ground-truth labels.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "ceas_clean.csv"
METRICS = ROOT / "results" / "metrics"
PRIVATE_AUDIT = ROOT / "data" / "audit"

METRICS.mkdir(parents=True, exist_ok=True)
PRIVATE_AUDIT.mkdir(parents=True, exist_ok=True)

print("=" * 58)
print("CYB-5 — EXPERIMENT 36")
print("CEAS DATASET READINESS AUDIT")
print("=" * 58)

# ------------------------------------------------------------
# 1. Load and validate the dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT)

required = {"source", "text", "label", "text_hash"}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

if df[["text", "label", "text_hash"]].isna().any().any():
    raise ValueError("Missing essential values detected.")

df["label"] = df["label"].astype(int)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected label values.")

print("\n=== DATASET STRUCTURE ===")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Labels:")
print(df["label"].value_counts().sort_index())

print("\n=== SOURCE DISTRIBUTION ===")
print(df["source"].value_counts(dropna=False).to_string())

print("\n=== SOURCE BY LABEL ===")
print(
    pd.crosstab(
        df["source"], df["label"]
    ).to_string()
)

print("\n=== TEXT HASH AUDIT ===")
print(
    "Repeated text_hash values:",
    int(df["text_hash"].duplicated().sum())
)

# ------------------------------------------------------------
# 2. Extract visible URLs
# ------------------------------------------------------------

URL_PATTERN = re.compile(
    r"""(?i)\b(?:https?://|www\.)[^\s<>"']+"""
)

def extract_urls(text):
    matches = URL_PATTERN.findall(str(text))
    urls = []

    for value in matches:
        cleaned = value.rstrip(".,;:!?)]}")

        if cleaned:
            urls.append(cleaned)

    return urls

df["urls_found"] = df["text"].map(extract_urls)
df["url_count"] = df["urls_found"].map(len)
df["has_url"] = df["url_count"] > 0

print("\n=== URL COVERAGE BY LABEL ===")

url_summary = (
    df.groupby("label")
    .agg(
        samples=("has_url", "size"),
        with_url=("has_url", "sum"),
        total_urls=("url_count", "sum"),
        mean_urls=("url_count", "mean"),
    )
    .reset_index()
)

url_summary["url_coverage"] = (
    url_summary["with_url"]
    / url_summary["samples"]
)

print(url_summary.round(4).to_string(index=False))

print("\nOverall emails containing a visible URL:")
print(
    int(df["has_url"].sum()),
    "/",
    len(df),
    f"({df['has_url'].mean():.2%})"
)

print("\n=== URL COUNT DISTRIBUTION ===")
print(
    df["url_count"].describe().round(2).to_string()
)

# ------------------------------------------------------------
# 3. Analyze URL syntax
# ------------------------------------------------------------

def url_host(raw_url):
    candidate = str(raw_url)

    if candidate.lower().startswith("www."):
        candidate = "http://" + candidate

    try:
        return (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        return ""

all_hosts = Counter()

for urls in df["urls_found"]:
    for url in urls:
        host = url_host(url)

        if host:
            all_hosts[host] += 1

print("\n=== URL PARSING ===")
print("Extracted URL occurrences:", int(df["url_count"].sum()))
print("Distinct parsed hosts:", len(all_hosts))

# Do not print actual URLs or hostnames:
# real email datasets may contain sensitive identifiers.

# ------------------------------------------------------------
# 4. Heuristic candidate indicators
# ------------------------------------------------------------

# This is a rough English-language screening vocabulary.
# It is NOT a verified phishing classification method.

INDICATORS = {
    "credential_request": (
        r"\b(?:password|login|log\s*in|sign\s*in|"
        r"credentials|verify your account)\b"
    ),
    "account_urgency": (
        r"\b(?:suspended|locked|urgent|"
        r"immediately|expire|expired)\b"
    ),
    "payment_request": (
        r"\b(?:payment|billing|invoice|"
        r"credit card|bank account)\b"
    ),
    "action_request": (
        r"\b(?:click here|confirm your|"
        r"update your|verify your)\b"
    ),
}

for name, pattern in INDICATORS.items():
    df[name] = df["text"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    )

indicator_columns = list(INDICATORS)

df["indicator_count"] = (
    df[indicator_columns].sum(axis=1)
)

print("\n=== HEURISTIC INDICATORS BY LABEL ===")

indicator_summary = (
    df.groupby("label")[
        indicator_columns
    ].sum()
)

print(indicator_summary.to_string())

# Candidate messages are only queued for review.
# Their true phishing status remains unverified.
df["review_candidate"] = (
    (df["label"] == 1)
    & (df["indicator_count"] >= 2)
)

print("\nPositive-label messages for manual review:")
print(int(df["review_candidate"].sum()))

print("Candidate messages containing visible URLs:")
print(
    int(
        (
            df["review_candidate"]
            & df["has_url"]
        ).sum()
    )
)

# ------------------------------------------------------------
# 5. Text length statistics
# ------------------------------------------------------------

df["character_length"] = df["text"].str.len()

print("\n=== TEXT LENGTH BY LABEL ===")

length_summary = (
    df.groupby("label")["character_length"]
    .describe()
)

print(length_summary.round(2).to_string())

# ------------------------------------------------------------
# 6. Save an aggregate report safe for GitHub
# ------------------------------------------------------------

url_summary.to_csv(
    METRICS / "ceas_url_coverage.csv",
    index=False
)

report = {
    "experiment": "CYB-5 Experiment 36",
    "dataset": "CEAS cleaned",
    "samples": int(len(df)),
    "label_distribution": {
        str(int(k)): int(v)
        for k, v in df["label"].value_counts().items()
    },
    "emails_with_visible_urls": int(df["has_url"].sum()),
    "url_coverage": float(df["has_url"].mean()),
    "url_occurrences": int(df["url_count"].sum()),
    "distinct_parsed_hosts": len(all_hosts),
    "repeated_text_hashes": int(
        df["text_hash"].duplicated().sum()
    ),
    "heuristic_review_candidates": int(
        df["review_candidate"].sum()
    ),
    "candidates_with_urls": int(
        (
            df["review_candidate"]
            & df["has_url"]
        ).sum()
    ),
    "limitations": [
        "Positive label is not verified phishing intent",
        "Keyword screening is not ground truth",
        "URL extraction covers visible http(s) and www links",
        "Obfuscated and bare-domain URLs may be missed",
        "Previously examined CEAS test is not untouched",
    ],
}

report_path = METRICS / "ceas_readiness_audit.json"

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 7. Save private review candidate identifiers
# ------------------------------------------------------------

# This file stays under /data/, which is gitignored.
# Never automatically publish raw email content.

candidate_file = (
    PRIVATE_AUDIT / "ceas_phishing_review_candidates.csv"
)

review_columns = [
    "source",
    "text_hash",
    "label",
    "url_count",
    "indicator_count",
] + indicator_columns

df.loc[
    df["review_candidate"],
    review_columns
].to_csv(
    candidate_file,
    index=False
)

print("\n=== EXPERIMENT 36 COMPLETED ===")
print("Aggregate JSON:", report_path)
print(
    "URL statistics:",
    METRICS / "ceas_url_coverage.csv"
)
print("Private candidate list:", candidate_file)
print("\nNO MODEL TRAINING OR SPLIT MODIFICATION")
