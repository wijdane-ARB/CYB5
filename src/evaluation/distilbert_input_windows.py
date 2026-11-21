
from pathlib import Path
import json
import zipfile
import warnings

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# CYB-5 — EXPERIMENT 32C
# DistilBERT Input Window Ablation
#
# Strategies:
#   1. head      : First 254 content tokens
#   2. tail      : Last 254 content tokens
#   3. head_tail : First 127 + Last 127 content tokens
#
# Model weights and decision threshold remain frozen.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "external" / "spaphish"
MODEL_DIR = ROOT / "models" / "distilbert_cyb5"

METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_RESULTS_ZIP = (
    ROOT / "results" / "CYB5_DistilBERT_Results.zip"
)

print("=" * 60)
print("CYB-5 — EXPERIMENT 32C")
print("DISTILBERT INPUT WINDOW ABLATION")
print("=" * 60)

# ============================================================
# 1. RELOAD FROZEN EXPERIMENT CONFIGURATION
# ============================================================

with zipfile.ZipFile(MODEL_RESULTS_ZIP) as archive:
    experiment = json.loads(
        archive.read("distilbert_metrics.json")
    )

THRESHOLD = float(experiment["test"]["threshold"])

MAX_LENGTH = 256

print("\n=== EXPERIMENT CONFIGURATION ===")
print("Frozen threshold:", THRESHOLD)
print("Maximum sequence length:", MAX_LENGTH)

# ============================================================
# 2. LOAD EXTERNAL SPAPHISH DATASET
# ============================================================

csv_files = list(DATA_DIR.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError(
        f"Expected exactly one SpaPhish CSV, found "
        f"{len(csv_files)}: {csv_files}"
    )

df = pd.read_csv(
    csv_files[0],
    sep=None,
    engine="python",
)

required_columns = {"subject", "body", "Label"}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing dataset columns: {missing}"
    )

texts = (
    df["subject"].fillna("").astype(str)
    + "\n"
    + df["body"].fillna("").astype(str)
).tolist()

labels = df["Label"].astype(int).to_numpy()

if set(np.unique(labels)) != {0, 1}:
    raise ValueError(
        "Unexpected SpaPhish class labels."
    )

print("\n=== DATASET ===")
print("File:", csv_files[0].name)
print("Total emails:", len(df))
print("SAFE:", int(np.sum(labels == 0)))
print("PHISHING:", int(np.sum(labels == 1)))

# ============================================================
# 3. LOAD FROZEN DISTILBERT
# ============================================================

print("\nLoading trained DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

if (
    model.config.id2label[0].upper() != "SAFE"
    or model.config.id2label[1].upper() != "PHISHING"
):
    raise ValueError(
        "Unexpected class mapping in trained model."
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)

# ============================================================
# 4. TOKENIZER CONFIGURATION
# ============================================================

SPECIAL_TOKENS = tokenizer.num_special_tokens_to_add(
    pair=False
)

CAPACITY = MAX_LENGTH - SPECIAL_TOKENS

CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id
PAD_ID = tokenizer.pad_token_id

if CLS_ID is None or SEP_ID is None or PAD_ID is None:
    raise ValueError(
        "Tokenizer is missing required special tokens."
    )

if SPECIAL_TOKENS != 2:
    raise ValueError(
        "This script expects a DistilBERT tokenizer "
        "with exactly two single-sequence special tokens."
    )

if CAPACITY <= 0:
    raise ValueError(
        "Invalid content-token capacity."
    )

print("\n=== TOKENIZER ===")
print("CLS token:", tokenizer.cls_token)
print("SEP token:", tokenizer.sep_token)
print("PAD token:", tokenizer.pad_token)
print("Special tokens:", SPECIAL_TOKENS)
print("Available content tokens:", CAPACITY)

# ============================================================
# 5. TOKENIZE EMAILS WITHOUT TRUNCATION
# ============================================================

print("\nTokenizing SpaPhish emails...")

encoded_texts = []

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Token indices sequence length is longer"
    )

    for text in texts:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]

        encoded_texts.append(token_ids)

token_lengths = np.array(
    [
        len(ids) + SPECIAL_TOKENS
        for ids in encoded_texts
    ],
    dtype=int,
)

long_mask = token_lengths > MAX_LENGTH

print("Total emails:", len(encoded_texts))
print("Emails exceeding 256 tokens:", int(long_mask.sum()))
print("Emails within 256 tokens:", int((~long_mask).sum()))

# Ensure consistency with Experiment 32A.
if int(long_mask.sum()) != 758:
    raise RuntimeError(
        "Token-length count differs from Experiment 32A. "
        "Check tokenizer and text preprocessing."
    )

# ============================================================
# 6. BUILD INPUT WINDOWS
# ============================================================

def choose_window(ids, strategy):
    """
    Construct one DistilBERT input sequence of at most
    MAX_LENGTH tokens.

    DistilBERT single-sequence format:
        [CLS] + content tokens + [SEP]

    For short emails, all strategies use identical tokens.
    """

    if len(ids) <= CAPACITY:
        selected = ids

    elif strategy == "head":
        selected = ids[:CAPACITY]

    elif strategy == "tail":
        selected = ids[-CAPACITY:]

    elif strategy == "head_tail":
        first_count = CAPACITY // 2
        last_count = CAPACITY - first_count

        selected = (
            ids[:first_count]
            + ids[-last_count:]
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    # Explicit special-token construction.
    # Avoids tokenizer.build_inputs_with_special_tokens()
    # incompatibility in the installed Transformers version.
    result = [CLS_ID] + list(selected) + [SEP_ID]

    if len(result) > MAX_LENGTH:
        raise RuntimeError(
            "Constructed sequence exceeds maximum length."
        )

    return result

# Check head equivalence against standard tokenization
# before running all predictions.
for text, ids in zip(texts[:20], encoded_texts[:20]):
    manual = choose_window(ids, "head")

    standard = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    )["input_ids"]

    if manual != standard:
        raise RuntimeError(
            "Manual head tokenization differs from "
            "standard DistilBERT tokenization."
        )

print("\nHead-window tokenization check passed.")

# ============================================================
# 7. FROZEN DISTILBERT INFERENCE
# ============================================================

BATCH_SIZE = 16 if device.type == "cuda" else 8

def predict(strategy):

    probabilities = []

    print(f"\nRunning strategy: {strategy}")

    with torch.inference_mode():

        for start in range(
            0,
            len(encoded_texts),
            BATCH_SIZE,
        ):

            current = encoded_texts[
                start:start + BATCH_SIZE
            ]

            sequences = [
                choose_window(ids, strategy)
                for ids in current
            ]

            max_batch_length = max(
                len(sequence) for sequence in sequences
            )

            padded = [
                sequence
                + [PAD_ID] * (
                    max_batch_length - len(sequence)
                )
                for sequence in sequences
            ]

            masks = [
                [1] * len(sequence)
                + [0] * (
                    max_batch_length - len(sequence)
                )
                for sequence in sequences
            ]

            inputs = {
                "input_ids": torch.tensor(
                    padded,
                    dtype=torch.long,
                    device=device,
                ),
                "attention_mask": torch.tensor(
                    masks,
                    dtype=torch.long,
                    device=device,
                ),
            }

            logits = model(**inputs).logits

            scores = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            probabilities.extend(
                scores.cpu().numpy().tolist()
            )

            if start % 160 == 0:
                print(
                    "Processed:",
                    min(start + len(current), len(texts)),
                    "/",
                    len(texts),
                )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return probabilities, predictions

# ============================================================
# 8. EVALUATE EACH INPUT STRATEGY
# ============================================================

strategies = [
    "head",
    "tail",
    "head_tail",
]

results = {}
summary_rows = []

groups = {
    "all": np.ones(
        len(labels),
        dtype=bool,
    ),
    "short_only": ~long_mask,
    "long_only": long_mask,
    "long_safe": long_mask & (labels == 0),
    "long_phishing": long_mask & (labels == 1),
}

for strategy in strategies:

    probabilities, predictions = predict(strategy)

    results[strategy] = {
        "probabilities": probabilities,
        "predictions": predictions,
    }

    for group_name, mask in groups.items():

        actual = labels[mask]
        predicted = predictions[mask]

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).ravel()

        total = len(actual)

        summary_rows.append({
            "strategy": strategy,
            "group": group_name,
            "samples": int(total),
            "accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "precision": float(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            "fpr": (
                float(fp / (fp + tn))
                if (fp + tn) else None
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    print(f"Strategy {strategy}: completed.")

# ============================================================
# 9. VERIFY EXPERIMENT 31 REPRODUCTION
# ============================================================

head_predictions = results["head"]["predictions"]

tn, fp, fn, tp = confusion_matrix(
    labels,
    head_predictions,
    labels=[0, 1],
).ravel()

observed = tuple(
    map(int, (tn, fp, fn, tp))
)

expected = (122, 542, 156, 575)

print("\n=== BASELINE REPRODUCTION ===")
print("Observed:", observed)
print("Expected:", expected)

if observed != expected:
    raise RuntimeError(
        "Head-window predictions do not reproduce "
        "Experiment 31. Stop before interpreting results."
    )

print("Experiment 31 reproduced successfully!")

# ============================================================
# 10. STRATEGY COMPARISON
# ============================================================

summary = pd.DataFrame(summary_rows)

print("\n=== INPUT WINDOW COMPARISON ===")
print(
    summary.round(4).to_string(index=False)
)

# ============================================================
# 11. PREDICTION FLIP ANALYSIS
# ============================================================

print("\n=== PREDICTION FLIPS ON LONG EMAILS ===")

flip_results = {}

for strategy in ["tail", "head_tail"]:

    alternative = results[strategy]["predictions"]

    flips = (
        alternative[long_mask]
        != head_predictions[long_mask]
    )

    total_flips = int(flips.sum())

    original_errors = (
        head_predictions[long_mask]
        != labels[long_mask]
    )

    alternative_errors = (
        alternative[long_mask]
        != labels[long_mask]
    )

    corrected = int(
        np.sum(
            original_errors
            & ~alternative_errors
        )
    )

    newly_wrong = int(
        np.sum(
            ~original_errors
            & alternative_errors
        )
    )

    flip_results[strategy] = {
        "flips": total_flips,
        "corrected_errors": corrected,
        "new_errors": newly_wrong,
    }

    print(
        f"\nStrategy: {strategy}"
    )
    print("Prediction flips:", total_flips)
    print("Previously incorrect, now correct:", corrected)
    print("Previously correct, now incorrect:", newly_wrong)

# ============================================================
# 12. EXPORT AGGREGATE RESULTS
# ============================================================

summary_path = (
    METRICS_DIR
    / "distilbert_input_windows.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

report = {
    "experiment": "CYB-5 Experiment 32C",
    "dataset": "SpaPhish",
    "model": "Frozen fine-tuned DistilBERT",
    "threshold": THRESHOLD,
    "max_length": MAX_LENGTH,
    "total_samples": len(labels),
    "long_emails": int(long_mask.sum()),
    "strategies": strategies,
    "flip_analysis": flip_results,
    "baseline_confusion_matrix": [
        [int(tn), int(fp)],
        [int(fn), int(tp)],
    ],
    "limitations": [
        "Exploratory input-window ablation",
        "Model weights remain frozen",
        "Decision threshold unchanged",
        "No causal attribution to truncation",
        "SpaPhish was previously inspected",
        "A different input window changes the input distribution",
    ],
}

json_path = (
    METRICS_DIR
    / "distilbert_input_windows.json"
)

json_path.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print("\n=== EXPERIMENT 32C COMPLETED ===")
print("CSV:", summary_path)
print("JSON:", json_path)