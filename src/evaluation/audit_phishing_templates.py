
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "phishing_validation"
OUT = ROOT / "reports"

emails = pd.read_csv(
    DATA / "Phishing_validation_emails_v3.csv"
)
metadata = pd.read_csv(
    DATA / "phishing_v3_metadata.csv"
)

if len(emails) != len(metadata):
    raise ValueError("Row count mismatch")

if not emails["Email Type"].reset_index(drop=True).equals(
    metadata["email_type"].reset_index(drop=True)
):
    raise ValueError("Label alignment mismatch")

df = metadata.copy()
df["label"] = emails["Email Type"].values

print("=== CYB-5 PHISHING TEMPLATE AUDIT ===")

print("\n=== SOURCE BY LABEL ===")
print(pd.crosstab(
    df["source_category"], df["label"]
).to_string())

print("\n=== TEMPLATE BY LABEL ===")
template_counts = pd.crosstab(
    df["template_id"].fillna("NO_TEMPLATE"),
    df["label"]
)

print("Total template groups:", len(template_counts))
print("Templates with both labels:",
      int((template_counts.gt(0).sum(axis=1) == 2).sum()))

print("\nTop 15 template groups:")
print(
    template_counts.assign(
        total=template_counts.sum(axis=1)
    ).sort_values("total", ascending=False)
    .head(15).to_string()
)

print("\n=== URL PRESENCE BY LABEL ===")
print(pd.crosstab(
    df["label"],
    df["has_url"],
    dropna=False
).to_string())

OUT.mkdir(parents=True, exist_ok=True)

template_counts.to_csv(
    OUT / "phishing_template_distribution.csv"
)

print("\n=== AUDIT COMPLETED ===")
