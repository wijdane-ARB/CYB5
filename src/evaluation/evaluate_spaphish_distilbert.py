
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "external" / "spaphish"
MODEL_DIR = ROOT / "models" / "distilbert_cyb5"
RESULTS_DIR = ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 EXPERIMENT 31 ===")
print("SPAPHISH EXTERNAL DISTILBERT EVALUATION")

# Load the original threshold saved during Experiment 30.
results_zip = (
    ROOT / "results" / "CYB5_DistilBERT_Results.zip"
)

with zipfile.ZipFile(results_zip) as archive:
    saved = json.loads(
        archive.read("distilbert_metrics.json")
    )

THRESHOLD = float(saved["test"]["threshold"])

print("\nFrozen threshold:", THRESHOLD)

# Read external corpus.
files = list(DATA_DIR.glob("*.csv"))

if len(files) != 1:
    raise ValueError(
        f"Expected one SpaPhish CSV, found {len(files)}"
    )

df = pd.read_csv(
    files[0],
    sep=None,
    engine="python"
)

df["text"] = (
    df["subject"].fillna("").astype(str)
    + "\n"
    + df["body"].fillna("").astype(str)
)

y_true = df["Label"].astype(int).to_numpy()

if set(np.unique(y_true)) != {0, 1}:
    raise ValueError("Unexpected label values")

print("Dataset:", files[0].name)
print("Emails:", len(df))
print("Safe:", int((y_true == 0).sum()))
print("Phishing:", int((y_true == 1).sum()))

# Load saved model without retraining.
print("\nLoading frozen DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

if (
    model.config.id2label[0].upper() != "SAFE"
    or model.config.id2label[1].upper() != "PHISHING"
):
    raise ValueError("Unexpected class mapping")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Device:", device)

# Match this limit to Experiment 30 training settings.
MAX_LENGTH = 256
BATCH_SIZE = 16 if device.type == "cuda" else 8

print("Maximum tokens:", MAX_LENGTH)

# Measure truncation before inference.
token_lengths = []

for text in df["text"].tolist():
    ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )["input_ids"]

    token_lengths.append(len(ids))

truncated_count = sum(
    length > MAX_LENGTH
    for length in token_lengths
)

print(
    "Emails exceeding token limit:",
    truncated_count,
    "/", len(df)
)

# Frozen model inference.
probabilities = []

print("\nStarting external inference...")

texts = df["text"].tolist()

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        logits = model(**encoded).logits

        scores = torch.softmax(
            logits, dim=1
        )[:, 1]

        probabilities.extend(
            scores.cpu().tolist()
        )

        if start % (BATCH_SIZE * 20) == 0:
            print(
                "Processed:",
                min(start + len(batch), len(texts)),
                "/",
                len(texts)
            )

probabilities = np.array(probabilities)

predictions = (
    probabilities >= THRESHOLD
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    predictions,
    labels=[0, 1]
).ravel()

metrics = {
    "dataset": "SpaPhish",
    "model": "Frozen DistilBERT",
    "samples": len(df),
    "threshold": THRESHOLD,
    "max_length": MAX_LENGTH,
    "truncated_emails": int(truncated_count),
    "accuracy": float(accuracy_score(
        y_true, predictions
    )),
    "precision": float(precision_score(
        y_true, predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_true, predictions, zero_division=0
    )),
    "f1": float(f1_score(
        y_true, predictions, zero_division=0
    )),
    "fpr": float(fp / (fp + tn)),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}

print("\n=== EXTERNAL EVALUATION RESULTS ===")

print(
    classification_report(
        y_true,
        predictions,
        target_names=["SAFE", "PHISHING"],
        zero_division=0
    )
)

print("Confusion matrix:")
print([[int(tn), int(fp)],
       [int(fn), int(tp)]])

for key, value in metrics.items():
    print(f"{key}: {value}")

# Save aggregate results.
output = (
    RESULTS_DIR
    / "spaphish_distilbert_external.json"
)

output.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

print("\n=== EXPERIMENT 31 COMPLETED ===")
print("Saved:", output)
