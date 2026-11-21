
from pathlib import Path
import json
import zipfile
import warnings

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "spaphish"
MODEL_DIR = ROOT / "models" / "distilbert_cyb5"
RESULTS_DIR = ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 EXPERIMENT 32A ===")
print("DISTILBERT ERROR AND TRUNCATION ANALYSIS")

# --------------------------------------------------
# 1. Load the frozen evaluation configuration
# --------------------------------------------------

with zipfile.ZipFile(
    ROOT / "results" / "CYB5_DistilBERT_Results.zip"
) as archive:
    saved = json.loads(
        archive.read("distilbert_metrics.json")
    )

THRESHOLD = float(saved["test"]["threshold"])
MAX_LENGTH = 256

print("\nFrozen threshold:", THRESHOLD)
print("Maximum input tokens:", MAX_LENGTH)

# --------------------------------------------------
# 2. Load SpaPhish without changing the dataset
# --------------------------------------------------

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError("Expected exactly one SpaPhish CSV.")

df = pd.read_csv(
    csv_files[0],
    sep=None,
    engine="python"
)

df["text"] = (
    df["subject"].fillna("").astype(str)
    + "\n"
    + df["body"].fillna("").astype(str)
)

df["label"] = df["Label"].astype(int)

if set(df["label"].unique()) != {0, 1}:
    raise ValueError("Unexpected label mapping.")

print("Emails:", len(df))
print("SAFE:", int((df.label == 0).sum()))
print("PHISHING:", int((df.label == 1).sum()))

# --------------------------------------------------
# 3. Load the existing trained DistilBERT
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

assert model.config.id2label[0].upper() == "SAFE"
assert model.config.id2label[1].upper() == "PHISHING"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Inference device:", device)

# --------------------------------------------------
# 4. Measure actual token lengths
# --------------------------------------------------

print("\nMeasuring token lengths...")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length is longer"
    )

    df["token_length"] = [
        len(
            tokenizer(
                text,
                truncation=False,
                add_special_tokens=True
            )["input_ids"]
        )
        for text in df["text"]
    ]

df["truncated"] = (
    df["token_length"] > MAX_LENGTH
)

print(
    "Emails exceeding 256 tokens:",
    int(df["truncated"].sum())
)

# --------------------------------------------------
# 5. Reproduce the original frozen predictions
# --------------------------------------------------

print("\nRunning frozen inference...")

BATCH_SIZE = 8 if device.type == "cpu" else 16
probabilities = []

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

        probabilities.extend(
            torch.softmax(logits, dim=1)[:, 1]
            .cpu().numpy().tolist()
        )

        if start % 160 == 0:
            print(
                f"Processed {min(start + len(batch), len(texts))}"
                f"/{len(texts)}"
            )

df["phishing_probability"] = probabilities

df["prediction"] = (
    df["phishing_probability"] >= THRESHOLD
).astype(int)

df["correct"] = df["prediction"] == df["label"]
df["error"] = ~df["correct"]

df["confidence"] = np.maximum(
    df["phishing_probability"],
    1 - df["phishing_probability"]
)

# --------------------------------------------------
# 6. Verify exact reproduction of Experiment 31
# --------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    df["label"],
    df["prediction"],
    labels=[0, 1]
).ravel()

print("\n=== CONFUSION MATRIX ===")
print([[int(tn), int(fp)], [int(fn), int(tp)]])

expected = (122, 542, 156, 575)
observed = tuple(map(int, (tn, fp, fn, tp)))

if observed != expected:
    raise RuntimeError(
        f"Predictions differ from Experiment 31: "
        f"{observed} != {expected}. "
        "Check preprocessing, model and tokenizer."
    )

print("Experiment 31 predictions reproduced!")

# --------------------------------------------------
# 7. Truncation versus prediction errors
# --------------------------------------------------

print("\n=== TRUNCATION ANALYSIS ===")

truncation_summary = (
    df.groupby("truncated")
    .agg(
        samples=("error", "size"),
        errors=("error", "sum"),
        error_rate=("error", "mean"),
        mean_confidence=("confidence", "mean")
    )
    .reset_index()
)

print(
    truncation_summary.round(4)
    .to_string(index=False)
)

# --------------------------------------------------
# 8. Errors by true class and truncation
# --------------------------------------------------

print("\n=== ERRORS BY CLASS AND TRUNCATION ===")

class_summary = (
    df.groupby(["label", "truncated"])
    .agg(
        samples=("error", "size"),
        errors=("error", "sum"),
        error_rate=("error", "mean"),
        mean_confidence=("confidence", "mean")
    )
    .reset_index()
)

print(
    class_summary.round(4).to_string(index=False)
)

# --------------------------------------------------
# 9. Errors by token-length category
# --------------------------------------------------

df["length_group"] = pd.cut(
    df["token_length"],
    bins=[0, 128, 256, 512, 1024, np.inf],
    labels=[
        "1-128",
        "129-256",
        "257-512",
        "513-1024",
        ">1024"
    ],
    include_lowest=True
)

print("\n=== ERRORS BY TOKEN LENGTH ===")

length_summary = (
    df.groupby(
        "length_group",
        observed=True
    )
    .agg(
        samples=("error", "size"),
        errors=("error", "sum"),
        error_rate=("error", "mean"),
        mean_confidence=("confidence", "mean")
    )
    .reset_index()
)

print(
    length_summary.round(4).to_string(index=False)
)

# --------------------------------------------------
# 10. Prediction confidence analysis
# --------------------------------------------------

print("\n=== CONFIDENCE ANALYSIS ===")

confidence_summary = (
    df.groupby("correct")
    .agg(
        samples=("confidence", "size"),
        mean_confidence=("confidence", "mean"),
        median_confidence=("confidence", "median")
    )
    .reset_index()
)

print(
    confidence_summary.round(4).to_string(index=False)
)

high_confidence_errors = int(
    (
        df["error"]
        & (df["confidence"] >= 0.90)
    ).sum()
)

print(
    "Incorrect predictions with confidence >= 0.90:",
    high_confidence_errors
)

# --------------------------------------------------
# 11. Export aggregate results only
# --------------------------------------------------

truncation_summary.to_csv(
    RESULTS_DIR / "distilbert_truncation_summary.csv",
    index=False
)

class_summary.to_csv(
    RESULTS_DIR / "distilbert_errors_by_class.csv",
    index=False
)

length_summary.to_csv(
    RESULTS_DIR / "distilbert_errors_by_length.csv",
    index=False
)

confidence_summary.to_csv(
    RESULTS_DIR / "distilbert_confidence_summary.csv",
    index=False
)

report = {
    "experiment": "CYB-5 Experiment 32A",
    "dataset": "SpaPhish",
    "model": "Frozen fine-tuned DistilBERT",
    "threshold": THRESHOLD,
    "max_length": MAX_LENGTH,
    "samples": int(len(df)),
    "correct": int(df["correct"].sum()),
    "errors": int(df["error"].sum()),
    "truncated": int(df["truncated"].sum()),
    "high_confidence_errors": high_confidence_errors,
    "confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)]
    ],
    "limitations": [
        "Spanish cross-domain corpus",
        "Observational association, not causation",
        "Model and threshold not modified",
        "Near-duplicate audit remains incomplete"
    ]
}

(
    RESULTS_DIR / "distilbert_error_analysis.json"
).write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\n=== EXPERIMENT 32A COMPLETED ===")
print("Results saved in:", RESULTS_DIR)
