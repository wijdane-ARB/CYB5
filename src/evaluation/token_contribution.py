
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "metrics"

model = joblib.load(
    ROOT / "models" / "baseline_tfidf_lr.joblib"
)

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]

nazario = pd.read_csv(DATA / "nazario_external.csv")
texts = nazario["text"].dropna().astype(str)

footer = "\n\nThis message is intended for its recipient."

print("=== CYB-5 TOKEN CONTRIBUTION ANALYSIS ===")

# Analyze the footer's individual vocabulary tokens.
tokens = vectorizer.build_analyzer()(footer)
vocab = vectorizer.vocabulary_
weights = classifier.coef_[0]

print("\nFooter tokens and classifier coefficients:")

for token in sorted(set(tokens)):
    idx = vocab.get(token)

    if idx is None:
        print(f"{token:25} NOT IN VOCABULARY")
    else:
        weight = weights[idx]
        direction = (
            "POSITIVE" if weight > 0
            else "NEGATIVE" if weight < 0
            else "NEUTRAL"
        )
        print(
            f"{token:25} {weight:+.4f} "
            f"{direction}"
        )

# Compare the actual model scores.
original_scores = model.decision_function(texts)
modified_scores = model.decision_function(texts + footer)

changes = modified_scores - original_scores

print("\n=== DECISION SCORE CHANGES ===")
print(pd.Series(changes).describe().round(4))

print("\nScores decreased:", int((changes < -1e-10).sum()))
print("Scores increased:", int((changes > 1e-10).sum()))
print("Scores unchanged:", int(
    np.isclose(changes, 0, atol=1e-10).sum()
))

print("\nMean score change:", round(changes.mean(), 4))

results = pd.DataFrame({
    "text_hash": nazario.loc[texts.index, "text_hash"].values,
    "original_score": original_scores,
    "modified_score": modified_scores,
    "score_change": changes
})

OUT.mkdir(parents=True, exist_ok=True)
results.to_csv(
    OUT / "token_contribution.csv",
    index=False
)

print("\nAnalysis completed.")
