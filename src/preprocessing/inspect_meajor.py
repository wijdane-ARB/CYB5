
from pathlib import Path
from collections import Counter
import json
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_benchmark"
REPORTS = ROOT / "reports"

PATH = DATA / "meajor_cleaned_preprocessed.parquet.gzip"

print("=== CYB-5 MEAJOR DATASET INSPECTION ===")

if not PATH.exists():
    raise FileNotFoundError(f"Dataset missing: {PATH}")

parquet_file = pq.ParquetFile(PATH)

columns = parquet_file.schema_arrow.names
total_rows = parquet_file.metadata.num_rows

print("\nTotal rows:", total_rows)

print("\n=== COLUMNS ===")
for field in parquet_file.schema_arrow:
    print(f"{field.name}: {field.type}")

def find_column(names):
    lookup = {c.lower(): c for c in columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None

label_col = find_column([
    "label", "class", "target", "binary_label"
])

source_col = find_column([
    "source", "dataset", "source_dataset",
    "dataset_source", "corpus"
])

text_col = find_column([
    "text", "body", "email_text",
    "content", "message", "email"
])

print("\n=== IDENTIFIED COLUMNS ===")
print("Label:", label_col)
print("Source:", source_col)
print("Text:", text_col)

label_counts = Counter()
source_counts = Counter()
missing_text = 0

selected = list(dict.fromkeys(
    c for c in [label_col, source_col, text_col]
    if c is not None
))

if selected:
    for batch in parquet_file.iter_batches(
        batch_size=10000,
        columns=selected
    ):
        records = batch.to_pydict()

        if label_col:
            label_counts.update(
                str(x) for x in records[label_col]
            )

        if source_col:
            source_counts.update(
                str(x) for x in records[source_col]
            )

        if text_col:
            missing_text += sum(
                x is None or not str(x).strip()
                for x in records[text_col]
            )

print("\n=== LABEL DISTRIBUTION ===")
print(dict(label_counts))

print("\n=== SOURCE DISTRIBUTION ===")
print(dict(source_counts))

print("\nMissing or empty text:", missing_text)

summary = {
    "dataset": "MeAJOR",
    "total_rows": total_rows,
    "columns": columns,
    "label_column": label_col,
    "source_column": source_col,
    "text_column": text_col,
    "label_distribution": dict(label_counts),
    "source_distribution": dict(source_counts),
    "missing_or_empty_text": missing_text
}

REPORTS.mkdir(parents=True, exist_ok=True)

output = REPORTS / "meajor_inspection.json"

output.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== INSPECTION COMPLETED ===")
print("Report saved:", output)
