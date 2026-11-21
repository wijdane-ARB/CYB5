
from pathlib import Path
from statistics import NormalDist
import json

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

print("=== CYB-5 EXPERIMENT 32B ===")
print("DISTILBERT STATISTICAL ERROR ANALYSIS")

# Wilson 95% confidence interval
def wilson_interval(errors, total, confidence=0.95):
    if total <= 0:
        return float("nan"), float("nan")

    z = NormalDist().inv_cdf(
        1 - (1 - confidence) / 2
    )

    p = errors / total
    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    margin = (
        z / denominator
        * (
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        ) ** 0.5
    )

    return max(0, center - margin), min(1, center + margin)


# Load aggregate results from Experiment 32A
df = pd.read_csv(
    METRICS / "distilbert_errors_by_class.csv"
)

required = {"label", "truncated", "samples", "errors"}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

if len(df) != 4:
    raise ValueError(
        "Expected four class-by-truncation groups"
    )

df["samples"] = df["samples"].astype(int)
df["errors"] = df["errors"].astype(int)

if (df["errors"] > df["samples"]).any():
    raise ValueError("Errors exceed sample counts")

if df["samples"].sum() != 1395:
    raise ValueError("Unexpected dataset size")

if df["errors"].sum() != 698:
    raise ValueError("Unexpected number of errors")


def boolean_value(value):
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


df["truncated"] = df["truncated"].map(boolean_value)

# Calculate confidence intervals
rows = []

for _, row in df.iterrows():
    n = int(row["samples"])
    errors = int(row["errors"])

    lower, upper = wilson_interval(errors, n)

    rows.append({
        "class": (
            "SAFE" if int(row["label"]) == 0
            else "PHISHING"
        ),
        "truncated": bool(row["truncated"]),
        "samples": n,
        "errors": errors,
        "error_rate": errors / n,
        "ci_lower": lower,
        "ci_upper": upper
    })

result = pd.DataFrame(rows)

print("\n=== ERROR RATES AND 95% CI ===")

print(
    result.round(4).to_string(index=False)
)

# Calculate within-class absolute differences.
print("\n=== TRUNCATED MINUS NON-TRUNCATED ===")

differences = {}

for label in ["SAFE", "PHISHING"]:
    subset = result[result["class"] == label]

    short = subset[subset["truncated"] == False].iloc[0]
    long = subset[subset["truncated"] == True].iloc[0]

    difference = (
        long["error_rate"] - short["error_rate"]
    )

    differences[label] = float(difference)

    print(
        f"{label}: {difference * 100:+.2f} "
        "percentage points"
    )

# Create publication-ready figure.
plot_df = result.copy()

plot_df["group"] = plot_df.apply(
    lambda r: (
        f"{r['class']} — "
        f"{'>256' if r['truncated'] else '≤256'} tokens"
    ),
    axis=1
)

fig, ax = plt.subplots(figsize=(10, 5.5))

x = list(range(len(plot_df)))

rates = (plot_df["error_rate"] * 100).to_numpy()
lower = (plot_df["ci_lower"] * 100).to_numpy()
upper = (plot_df["ci_upper"] * 100).to_numpy()

ax.bar(x, rates)

ax.errorbar(
    x,
    rates,
    yerr=[rates - lower, upper - rates],
    fmt="none",
    capsize=5,
    ecolor="black"
)

ax.set_xticks(x)
ax.set_xticklabels(
    plot_df["group"],
    rotation=12,
    ha="right"
)

ax.set_ylabel("Error rate (%)")
ax.set_ylim(0, 110)

ax.set_title(
    "DistilBERT on SpaPhish: Error Rates by Class "
    "and Token-Length Group"
)

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

figure_path = (
    FIGURES / "distilbert_error_wilson_ci.png"
)

fig.savefig(figure_path, dpi=300)
plt.close(fig)

# Export results
csv_path = (
    METRICS / "distilbert_error_confidence_intervals.csv"
)

result.to_csv(csv_path, index=False)

json_path = (
    METRICS / "distilbert_statistical_analysis.json"
)

json_path.write_text(
    json.dumps({
        "experiment": "CYB-5 32B",
        "confidence_level": 0.95,
        "method": "Wilson score interval",
        "observed_differences": differences,
        "limitations": [
            "Observational analysis",
            "No causal attribution to truncation",
            "Independence assumption may be imperfect",
            "Spanish-language external corpus"
        ]
    }, indent=2),
    encoding="utf-8"
)

print("\n=== EXPERIMENT 32B COMPLETED ===")
print("CSV:", csv_path)
print("Figure:", figure_path)
print("JSON:", json_path)
