
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "phishing_benchmark"
    / "meajor_cleaned_preprocessed.parquet.gzip"
)

OUTPUT = ROOT / "reports" / "meajor_quality_audit.json"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

counts = Counter()
source_labels = Counter()
text_hashes = defaultdict(set)
hash_counts = Counter()

missing = 0
short = 0

parquet = pq.ParquetFile(INPUT)

for batch in parquet.iter_batches(
    batch_size=5000,
    columns=["subject", "body", "source", "label"]
):
    rows = batch.to_pylist()

    for row in rows:
        source = row["source"]
        label = row["label"]

        if (
            source is None
            or label is None
            or row["body"] is None
        ):
            missing += 1
            continue

        label = int(label)

        if label not in (0, 1):
            raise ValueError(f"Unexpected label: {label}")

        subject = row["subject"] or ""
        body = row["body"] or ""

        text = (subject + "\n" + body).strip()

        if len(text) < 20:
            short += 1
            continue

        # Match messages despite case/whitespace differences.
        normalized = " ".join(text.lower().split())

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        counts[source] += 1
        source_labels[(source, label)] += 1

        hash_counts[digest] += 1
        text_hashes[digest].add(label)

duplicates = sum(
    count - 1
    for count in hash_counts.values()
    if count > 1
)

conflicts = sum(
    len(labels) > 1
    for labels in text_hashes.values()
)

print("=== CYB-5 MEAJOR QUALITY AUDIT ===")

print("\nValid source counts:")
for source, count in sorted(counts.items()):
    print(source, count)

print("\n=== SOURCE / LABEL DISTRIBUTION ===")
for (source, label), count in sorted(source_labels.items()):
    print(f"{source} | label {label}: {count}")

print("\nMissing critical fields:", missing)
print("Short messages:", short)
print("Duplicate normalized texts:", duplicates)
print("Conflicting-label texts:", conflicts)

summary = {
    "valid_source_counts": dict(counts),
    "source_label_counts": {
        f"{source}_label_{label}": count
        for (source, label), count in source_labels.items()
    },
    "missing_critical_fields": missing,
    "short_messages": short,
    "duplicate_normalized_texts": duplicates,
    "conflicting_label_texts": conflicts
}

OUTPUT.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== AUDIT COMPLETED ===")
print("Saved:", OUTPUT)
