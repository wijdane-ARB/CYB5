
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_validation" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"

test = pd.read_parquet(DATA / "test.parquet")

model = joblib.load(
    MODELS / "phishing_template_lr.joblib"
)

THRESHOLD = 0.44

test["probability"] = model.predict_proba(
    test["text"]
)[:, 1]

test["prediction"] = (
    test["probability"] >= THRESHOLD
).astype(int)

test["correct"] = (
    test["prediction"] == test["label"]
)

test["text_length"] = (
    test["text"].astype(str).str.len()
)

summary = test.groupby(
    ["template_id", "label"]
).agg(
    samples=("label", "size"),
    accuracy=("correct", "mean"),
    mean_probability=("probability", "mean"),
    min_probability=("probability", "min"),
    max_probability=("probability", "max"),
    mean_length=("text_length", "mean")
).reset_index()

print("=== CYB-5 TEMPLATE FAILURE ANALYSIS ===")

print(
    summary.round(4).to_string(index=False)
)

print("\n=== SAFE_ORDER_CONF DETAILS ===")

problem = test[
    test["template_id"] == "SAFE_ORDER_CONF"
]

print("Samples:", len(problem))
print("Predicted phishing:",
      int((problem["prediction"] == 1).sum()))

print("\nProbability distribution:")
print(
    problem["probability"].describe().round(4)
)

print("\nAverage message length:",
      round(problem["text_length"].mean(), 2))

# Save aggregate results without raw email text.
summary.to_csv(
    METRICS / "phishing_template_diagnosis.csv",
    index=False
)

print("\n=== DIAGNOSIS COMPLETED ===")
