"""Train URL-only baselines (LR deterministic; RandomForest with seeds 42/123/2026).
Threshold chosen on VALIDATION only. No network access. No email text is saved."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from url_features import (FEATURE_FAMILIES, FEATURE_NAMES, LABEL_CANDIDATES, TEXT_CANDIDATES,
                          binary_metrics, featurize, guess_column, normalize_labels, read_table)

METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "lr":
        return Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(solver="liblinear", class_weight="balanced",
                                                    max_iter=1000, random_state=seed))])
    return Pipeline([("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                    class_weight="balanced", random_state=seed,
                                                    n_jobs=1))])


def _find(d: Path, name: str):
    for ext in (".parquet", ".csv"):
        if (d / f"{name}{ext}").is_file():
            return d / f"{name}{ext}"
    return None


def load_synthetic(d: Path, tcol, lcol) -> dict:
    out = {}
    for name in ("train", "validation", "test"):
        p = _find(d, name)
        if p is None:
            if name == "test":
                continue
            raise FileNotFoundError(f"{name}.parquet/.csv not found in {d}")
        df = read_table(p)
        t = guess_column(df, tcol, TEXT_CANDIDATES, "text")
        l = guess_column(df, lcol, LABEL_CANDIDATES, "label")
        out[name] = pd.DataFrame({"text": df[t].astype(str), "label": normalize_labels(df[l])})
    return out


def load_ceas_group() -> dict:
    ceas = pd.read_csv(ROOT / "data" / "processed" / "ceas_clean.csv").dropna(
        subset=["text", "label", "text_hash"])
    out = {}
    for name in ("train", "validation", "test"):
        ids = pd.read_csv(METRICS / f"group_{name}_ids.csv")
        s = ceas.merge(ids[["text_hash"]], on="text_hash", how="inner", validate="one_to_one")
        if len(s) != len(ids):
            raise ValueError(f"{name}: split reconstruction mismatch")
        out[name] = pd.DataFrame({"text": s["text"].astype(str), "label": s["label"].astype(int)})
    return out


def coverage(name, df, X) -> dict:
    has = X[:, 0] > 0
    y = df["label"].to_numpy()
    info = {"messages": int(len(df)), "with_url": int(has.sum()), "coverage": float(has.mean())}
    for lab in (0, 1):
        m = y == lab
        info[f"label{lab}_n"] = int(m.sum())
        info[f"label{lab}_with_url"] = int(has[m].sum())
    print(f"[coverage:{name}] {info}")
    return info


def choose_threshold(y, p, rule: str, max_fpr: float):
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        m = binary_metrics(y, (p >= t).astype(int)); m["threshold"] = float(t); rows.append(m)
    tbl = pd.DataFrame(rows)
    if rule == "max_f1":
        tbl["dist"] = (tbl["threshold"] - 0.5).abs()
        best = tbl.sort_values(["f1", "dist"], ascending=[False, True]).iloc[0]
    else:
        ok = tbl[tbl["fpr"] <= max_fpr]
        if ok.empty:
            raise ValueError("No threshold satisfies the FPR constraint on validation")
        best = ok.sort_values(["recall", "fpr", "threshold"], ascending=[False, True, False]).iloc[0]
    return float(best["threshold"]), tbl


def run_ablation(Xtr, ytr, Xva, yva, rule, max_fpr) -> pd.DataFrame:
    rows = []

    def fit_eval(label, cols):
        idx = [FEATURE_NAMES.index(c) for c in cols]
        model = make_model("lr", 42).fit(Xtr[:, idx], ytr)
        p = model.predict_proba(Xva[:, idx])[:, 1]
        thr, _ = choose_threshold(yva, p, rule, max_fpr)
        m = binary_metrics(yva, (p >= thr).astype(int))
        rows.append({"variant": label, "n_features": len(idx), "threshold": thr,
                     "val_auc": float(roc_auc_score(yva, p)), "val_f1": m["f1"],
                     "val_recall": m["recall"], "val_fpr": m["fpr"]})

    fit_eval("all_features", FEATURE_NAMES)
    for fam, cols in FEATURE_FAMILIES.items():
        fit_eval(f"minus_{fam}", [c for c in FEATURE_NAMES if c not in cols])
        fit_eval(f"only_{fam}", cols)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-source", choices=["synthetic", "ceas_group"], required=True)
    ap.add_argument("--synthetic-dir", type=Path)
    ap.add_argument("--text-col"); ap.add_argument("--label-col")
    ap.add_argument("--models", nargs="+", choices=["lr", "rf"], default=["lr", "rf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    ap.add_argument("--threshold-rule", choices=["max_f1", "max_recall_at_fpr"], default="max_f1")
    ap.add_argument("--max-fpr", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models" / "url")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--eval-test", action="store_true",
                    help="Evaluate frozen models+thresholds on the held-out test (do this once, last)")
    ap.add_argument("--allow-low-coverage", action="store_true")
    a = ap.parse_args()

    if a.train_source == "synthetic":
        if not a.synthetic_dir:
            ap.error("--synthetic-dir is required with --train-source synthetic")
        splits = load_synthetic(a.synthetic_dir, a.text_col, a.label_col)
    else:
        splits = load_ceas_group()

    feats = {k: featurize(v["text"].tolist()) for k, v in splits.items()}
    cov = {k: coverage(k, splits[k], feats[k]) for k in splits}
    Xtr, ytr = feats["train"], splits["train"]["label"].to_numpy()
    Xva, yva = feats["validation"], splits["validation"]["label"].to_numpy()
    if len(set(ytr)) < 2 or len(set(yva)) < 2:
        raise SystemExit("Train and validation must contain both classes.")

    c = cov["train"]
    insufficient = (c["coverage"] < 0.05 or c["label0_with_url"] < 10 or c["label1_with_url"] < 10)
    if insufficient and not a.allow_low_coverage:
        raise SystemExit(
            "INSUFFICIENT URL SIGNAL in the training split (see coverage above). "
            "Declare it in the report; choose another documented train/validation "
            "(e.g. --train-source ceas_group) WITHOUT mixing in the test, or use --allow-low-coverage.")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "CYB-5 Exp42 URL-only", "train_source": a.train_source,
              "threshold_rule": a.threshold_rule, "max_fpr": a.max_fpr,
              "coverage": cov, "models": [],
              "notes": ["Threshold selected on validation only.",
                        "LR is deterministic (liblinear): no seed variation; trained once.",
                        "RandomForest repeated with seeds; std is over training seeds only."]}

    for kind in a.models:
        for seed in ([42] if kind == "lr" else a.seeds):
            model = make_model(kind, seed).fit(Xtr, ytr)
            pva = model.predict_proba(Xva)[:, 1]
            thr, _ = choose_threshold(yva, pva, a.threshold_rule, a.max_fpr)
            val = binary_metrics(yva, (pva >= thr).astype(int))
            val["auc"] = float(roc_auc_score(yva, pva))
            entry = {"name": f"URL_{kind.upper()}", "seed": seed, "threshold": thr, "validation": val}
            if a.eval_test and "test" in splits:
                yte = splits["test"]["label"].to_numpy()
                pte = model.predict_proba(feats["test"])[:, 1]
                entry["test"] = binary_metrics(yte, (pte >= thr).astype(int))
                entry["test"]["auc"] = float(roc_auc_score(yte, pte))
            joblib.dump({"name": f"URL_{kind.upper()}", "kind": kind, "seed": seed,
                         "pipeline": model, "feature_names": FEATURE_NAMES, "threshold": thr,
                         "threshold_rule": a.threshold_rule, "train_source": a.train_source,
                         "train_size": int(len(ytr))},
                        a.out_dir / f"url_{kind}_seed{seed}.joblib")
            report["models"].append(entry)
            print(f"URL_{kind.upper()} seed={seed} thr={thr:.2f} "
                  f"VAL F1={val['f1']:.4f} recall={val['recall']:.4f} FPR={val['fpr']:.4f} AUC={val['auc']:.4f}")
            if "test" in entry:
                t = entry["test"]
                print(f"   TEST F1={t['f1']:.4f} recall={t['recall']:.4f} FPR={t['fpr']:.4f}")

    (METRICS / "url_only_training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if a.ablation:
        abl = run_ablation(Xtr, ytr, Xva, yva, a.threshold_rule, a.max_fpr)
        abl.to_csv(METRICS / "url_only_ablation_validation.csv", index=False)
        print("\n=== ABLATION (LR, VALIDATION only; test not used) ===")
        print(abl.round(4).to_string(index=False))
    print("\nModels saved in:", a.out_dir, "| report:", METRICS / "url_only_training.json")


if __name__ == "__main__":
    main()