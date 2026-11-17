
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "results" / "metrics"
DATA = ROOT / "data" / "processed"

ceas = pd.read_csv(DATA / "ceas_clean.csv")

def load_split(name):
    split = pd.read_csv(
        METRICS / f"{name}_split_ids.csv"
    )
    return ceas.merge(
        split[["text_hash"]].drop_duplicates(),
        on="text_hash",
        how="inner"
    )

train = load_split("train")
validation = load_split("validation")
test = load_split("test")

print("=== CYB-5 SPLIT AUDIT ===")

for name, df in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(f"{name}: {len(df)} samples")

def normalize(text):
    return " ".join(str(text).lower().split())

sets = {}

for name, df in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    sets[name] = set(df["text"].map(normalize))

for a, b in [
    ("Train", "Validation"),
    ("Train", "Test"),
    ("Validation", "Test")
]:
    overlap = len(sets[a] & sets[b])
    print(f"{a} / {b} exact normalized overlap: {overlap}")

print("\n=== BASIC SOURCE ARTIFACT AUDIT ===")

for name, df in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(f"\n{name}")
    for label in [0, 1]:
        subset = df[df["label"] == label]
        print(
            f"Label {label}: {len(subset)} samples, "
            f"mean length: {subset['text'].str.len().mean():.0f}"
        )

print("\nAudit completed.")
