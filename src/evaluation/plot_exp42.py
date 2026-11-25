"""Figures for Exp42 (aggregates only, no email content)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "metrics" / "exp42_classical"
FIG = ROOT / "results" / "figures" / "exp42"
FIG.mkdir(parents=True, exist_ok=True)
PROV_PATH = SRC / "exp42_provenance.json"
if PROV_PATH.exists():
    _prov = json.loads(PROV_PATH.read_text(encoding="utf-8"))
    TAG = _prov.get("plot_title", "Exploratory pilot")
else:
    TAG = "Exploratory pilot"
ORDER = ["clean", "lexical_proxy", "neutral_hidden_html", "url_defang"]


def disp(df):
    return df["model"] + np.where(df["leakage_flag"], " [LEAKY]", "")


def main():
    agg = pd.read_csv(SRC / "exp42_aggregate.csv")
    runs = pd.read_csv(SRC / "exp42_metrics_by_run.csv")
    agg["display"], runs["display"] = disp(agg), disp(runs)

    for subset in agg["subset"].unique():
        a = agg[agg.subset == subset]
        cols = [c for c in ORDER if c in set(a.condition)]
        m = a.pivot(index="display", columns="condition", values="f1_mean")[cols]
        e = a.pivot(index="display", columns="condition", values="f1_std")[cols]
        ax = m.plot.bar(yerr=e, capsize=3, figsize=(9, 5))
        ax.set_ylabel("F1 (phishing)"); ax.set_ylim(0, 1.05); plt.xticks(rotation=20, ha="right")
        ax.set_title(f"F1 clean vs attacked - subset: {subset}\n{TAG}", fontsize=10)
        plt.tight_layout(); plt.savefig(FIG / f"f1_{subset}.png", dpi=150); plt.close()

        c = a[a.condition == "clean"].set_index("display")[["recall_mean", "fpr_mean"]]
        ax = c.plot.bar(figsize=(8, 4.5)); ax.set_ylim(0, 1.05)
        ax.set_title(f"Recall and FPR (clean) - subset: {subset}\n{TAG}", fontsize=10)
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(FIG / f"recall_fpr_{subset}.png", dpi=150); plt.close()

    c = runs[(runs.condition == "clean") & (runs.subset == "all")]
    c = c.sort_values("seed").groupby("display").first()
    fig, axes = plt.subplots(1, len(c), figsize=(3.4 * len(c), 3.4), squeeze=False)
    for ax, (name, r) in zip(axes[0], c.iterrows()):
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(name, fontsize=9)
    title = TAG
    fig.suptitle(f"Confusion matrices (clean)\n{title}", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "confusion_clean.png", dpi=150); plt.close()

    lr = sorted((ROOT / "models" / "url").glob("url_lr_seed*.joblib"))
    rf = sorted((ROOT / "models" / "url").glob("url_rf_seed42.joblib"))
    panels = []
    if lr:
        b = joblib.load(lr[0]); panels.append(("LR |coef| (standardized)", b["feature_names"],
                                              np.abs(b["pipeline"].named_steps["clf"].coef_[0])))
    if rf:
        b = joblib.load(rf[0]); panels.append(("RF importance (seed 42)", b["feature_names"],
                                              b["pipeline"].named_steps["clf"].feature_importances_))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), squeeze=False)
        for ax, (title, names, vals) in zip(axes[0], panels):
            o = np.argsort(vals)[-15:]
            ax.barh(np.array(names)[o], vals[o]); ax.set_title(title)
        plt.tight_layout(); plt.savefig(FIG / "url_feature_importance.png", dpi=150); plt.close()
    print("Figures saved in:", FIG)


if __name__ == "__main__":
    main()