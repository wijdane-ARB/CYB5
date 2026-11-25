"""Pilote Exp42 CEAS-only : 40 positifs + 40 négatifs, même source = pas de confound."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "data" / "audit" / "cyb5_experiment38_pilot"
OUT.mkdir(parents=True, exist_ok=True)

ceas = pd.read_csv(DATA / "ceas_clean.csv").drop_duplicates("text_hash")

pos = ceas[ceas.label == 1].sample(40, random_state=42)
neg = ceas[ceas.label == 0].sample(40, random_state=42)

pilot = pd.concat([pos, neg], ignore_index=True)
pilot["sample_id"] = [f"P{i:03d}" for i in range(len(pilot))]
pilot["source"] = "CEAS_08"  # source unique = pas de confound

out = OUT / "pilot_manifest_ceas_only.csv"
pilot[["sample_id", "text", "label", "source"]].to_csv(out, index=False)

# Vérification
from sys import path
path.insert(0, str(ROOT / "src" / "models"))
from url_features import URL_RE
pilot["has_url"] = pilot.text.astype(str).apply(lambda t: bool(URL_RE.search(t)))
print(f"Positifs : {(pilot.label==1).sum()} | avec URL : {pilot[pilot.label==1].has_url.sum()}")
print(f"Négatifs : {(pilot.label==0).sum()} | avec URL : {pilot[pilot.label==0].has_url.sum()}")
print(f"Écrit : {out}")