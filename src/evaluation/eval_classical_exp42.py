"""Exp42 adapter: URL-only + existing TF-IDF on the frozen pilot manifest (same attacks as Exp38).
Public outputs: aggregates only. Per-message predictions (IDs, no text) stay under data/audit/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from url_features import (
    FEATURE_NAMES,
    LABEL_CANDIDATES,
    TEXT_CANDIDATES,
    binary_metrics,
    featurize,
    guess_column,
    norm_hash,
    read_table,
    wilson,
)
import experiment38_distilbert_pilot as exp38  # reuse the FROZEN attacks and URL regex

ATTACKS, URL_RE = exp38.ATTACKS, exp38.URL_RE
OUT_PUBLIC = ROOT / "results" / "metrics" / "exp42_classical"
OUT_PRIVATE = ROOT / "data" / "audit" / "cyb5_experiment42"


def url_scorer(bundle):
    pipe = bundle["pipeline"]
    assert bundle["feature_names"] == FEATURE_NAMES, "feature list changed since training"
    pos = list(pipe.classes_).index(1)
    return lambda t: float(pipe.predict_proba(featurize([t]))[0, pos])


def tfidf_scorer(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        for k in ("pipeline", "model", "clf"):
            if k in obj:
                obj = obj[k]
                break
    if not hasattr(obj, "predict_proba"):
        raise TypeError(
            f"{path}: object has no predict_proba (type={type(obj)}). "
            "Open train_phishing_templates.py to see what is saved."
        )
    pos = list(obj.classes_).index(1)
    return lambda t: float(obj.predict_proba([t])[0, pos])


def score_all(fn, texts):
    fn(texts[0])  # warm-up
    probs, lat = np.empty(len(texts)), np.empty(len(texts))
    for i, t in enumerate(texts):
        s = perf_counter()
        probs[i] = fn(t)
        lat[i] = (perf_counter() - s) * 1000
    return probs, lat


def leakage_check(df, syn_dir):
    if syn_dir is None:
        return None
    pilot = {norm_hash(t) for t in df["text"]}
    res = {}
    for name in ("train", "validation", "test"):
        for ext in (".parquet", ".csv"):
            p = Path(syn_dir) / f"{name}{ext}"
            if p.is_file():
                d = read_table(p)
                col = guess_column(d, None, TEXT_CANDIDATES, "text")
                res[name] = len(pilot & {norm_hash(t) for t in d[col]})
                break
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/audit/cyb5_experiment38_pilot/pilot_manifest.csv",
    )
    ap.add_argument(
        "--url-models",
        nargs="+",
        type=Path,
        default=None,
        help="default: all models/url/*.joblib",
    )
    ap.add_argument(
        "--tfidf",
        nargs=4,
        action="append",
        default=[],
        metavar=("NAME", "PATH", "THRESHOLD", "TRAINED_ON"),
        help="TRAINED_ON = synthetic | ceas  (ceas => flagged LEAKY on this pilot)",
    )
    ap.add_argument("--synthetic-dir", type=Path, help="for exact-overlap check")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest, keep_default_na=False)
    for c in ("sample_id", "text", "label", "source"):
        if c not in df:
            raise ValueError(f"manifest missing {c}")
    if df.sample_id.duplicated().any():
        raise ValueError("duplicate sample_id")
    texts = df.text.astype(str).tolist()
    y = df.label.astype(int).to_numpy()
    pos, n = y == 1, len(df)
    has_url = np.array([bool(URL_RE.search(t)) for t in texts])
    print(
        f"Pilot: {n} messages | positives {int(pos.sum())} (with URL {int((pos & has_url).sum())}) "
        f"| negatives {int((~pos).sum())} (with URL {int((~pos & has_url).sum())})"
    )

    # ---- frozen variants (identical logic to Exp38) ----
    variants = {"clean": dict(texts=texts, eligible=np.zeros(n, bool), changed=np.zeros(n, bool))}
    for an, fn in ATTACKS.items():
        elig = (has_url if an == "url_defang" else np.ones(n, bool)) & pos
        mod = [fn(t) if e else t for t, e in zip(texts, elig)]
        changed = np.array([x != z for x, z in zip(texts, mod)])
        variants[an] = dict(texts=mod, eligible=elig & changed, changed=changed)

    # ---- models ----
    specs = []
    url_paths = a.url_models or sorted((ROOT / "models" / "url").glob("*.joblib"))
    for p in url_paths:
        b = joblib.load(p)
        specs.append(
            dict(
                name=b["name"],
                version=f"{b['name']}_v1_{b['train_source']}",
                seed=str(b["seed"]),
                fn=url_scorer(b),
                thr=float(b["threshold"]),
                leaky=(b["train_source"] == "ceas_group"),
                is_url=True,
            )
        )
    for name, path, thr, origin in a.tfidf:
        specs.append(
            dict(
                name=name,
                version=f"{name}_tfidf",
                seed="not_recorded",
                fn=tfidf_scorer(Path(path)),
                thr=float(thr),
                leaky=(origin.lower() == "ceas"),
                is_url=False,
            )
        )
    if not specs:
        raise SystemExit("No model given. Train URL models first and/or pass --tfidf.")

    subsets = {"all": np.ones(n, bool), "url_in_clean_text": has_url}
    runs, preds, extra = [], [], {"url_score_max_abs_diff": {}}
    for sp in specs:
        probs, lats, pr = {}, {}, {}
        for cond, v in variants.items():
            probs[cond], lats[cond] = score_all(sp["fn"], v["texts"])
            pr[cond] = (probs[cond] >= sp["thr"]).astype(int)
            preds.append(
                pd.DataFrame(
                    {
                        "model": sp["name"],
                        "model_version": sp["version"],
                        "seed": sp["seed"],
                        "sample_id": df.sample_id,
                        "condition": cond,
                        "source": df.source,
                        "label": y,
                        "pred": pr[cond],
                        "score": np.round(probs[cond], 6),
                        "eligible": v["eligible"],
                        "text_changed": v["changed"],
                        "latency_ms": np.round(lats[cond], 4),
                    }
                )
            )
        if sp["is_url"]:
            extra["url_score_max_abs_diff"][f"{sp['name']}_seed{sp['seed']}"] = {
                c: float(np.max(np.abs(probs[c] - probs["clean"])))
                for c in ("lexical_proxy", "neutral_hidden_html")
            }
        clean_f1 = {k: binary_metrics(y[m], pr["clean"][m])["f1"] for k, m in subsets.items()}
        for cond, v in variants.items():
            for sname, m in subsets.items():
                met = binary_metrics(y[m], pr[cond][m])
                orig = pos & m & v["eligible"] & (pr["clean"] == 1)
                flips = orig & (pr[cond] == 0)
                npos = int((pos & m).sum())
                r_lo, r_hi = wilson(met["tp"], met["tp"] + met["fn"])
                f_lo, f_hi = wilson(met["fp"], met["fp"] + met["tn"])
                a_lo, a_hi = wilson(int(flips.sum()), int(orig.sum()))
                runs.append(
                    {
                        "model": sp["name"],
                        "model_version": sp["version"],
                        "seed": sp["seed"],
                        "condition": cond,
                        "subset": sname,
                        "threshold": sp["thr"],
                        "leakage_flag": sp["leaky"],
                        **met,
                        "n_pos": npos,
                        "n_neg": int(m.sum()) - npos,
                        "f1_delta": met["f1"] - clean_f1[sname],
                        "asr": flips.sum() / orig.sum() if orig.sum() else np.nan,
                        "asr_flips": int(flips.sum()),
                        "asr_denominator": int(orig.sum()),
                        "asr_ci_lo": a_lo,
                        "asr_ci_hi": a_hi,
                        "coverage_eligible": (v["eligible"] & m).sum() / npos
                        if (npos and cond != "clean")
                        else np.nan,
                        "coverage_changed": ((v["changed"] & pos & m).sum() / npos)
                        if (npos and cond != "clean")
                        else np.nan,
                        "recall_ci_lo": r_lo,
                        "recall_ci_hi": r_hi,
                        "fpr_ci_lo": f_lo,
                        "fpr_ci_hi": f_hi,
                        "lat_mean_ms": float(lats[cond].mean()),
                        "lat_median_ms": float(np.median(lats[cond])),
                        "lat_p95_ms": float(np.percentile(lats[cond], 95)),
                    }
                )

    by_run = pd.DataFrame(runs)
    keys = ["model", "condition", "subset", "leakage_flag"]
    agg = (
        by_run.groupby(keys)
        .agg(
            n_runs=("seed", "size"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
            asr_mean=("asr", "mean"),
            asr_den=("asr_denominator", "mean"),
            coverage_eligible=("coverage_eligible", "mean"),
            lat_median_ms=("lat_median_ms", "mean"),
            lat_p95_ms=("lat_p95_ms", "mean"),
        )
        .reset_index()
    )
    for c in ("f1_std", "recall_std", "fpr_std"):
        agg[c] = agg[c].fillna(0.0)

    n_sources = int(df["source"].nunique()) if "source" in df else 1
    status = (
        "EXPLORATORY_PILOT_CORPUS_LABELS_SOURCE_CONFOUNDED"
        if n_sources > 1
        else "PILOT_SINGLE_SOURCE_BALANCED"
    )
    n_runs = (
        int(by_run[by_run.model.str.startswith("URL_RF")].seed.nunique())
        if len(by_run) and by_run.model.str.startswith("URL_RF").any()
        else (int(by_run.seed.nunique()) if len(by_run) else 1)
    )
    plot_title = (
        "Exploratory pilot - corpus labels, source-confounded"
        if n_sources > 1
        else "Pilot single-source balanced"
    )

    OUT_PUBLIC.mkdir(parents=True, exist_ok=True)
    OUT_PRIVATE.mkdir(parents=True, exist_ok=True)
    by_run.to_csv(OUT_PUBLIC / "exp42_metrics_by_run.csv", index=False)
    agg.to_csv(OUT_PUBLIC / "exp42_aggregate.csv", index=False)
    pd.concat(preds).to_csv(OUT_PRIVATE / "exp42_predictions_private.csv", index=False)
    extra.update(
        {
            "status": status,
            "plot_title": plot_title,
            "n_sources": n_sources,
            "n_messages": n,
            "positives_with_url": int((pos & has_url).sum()),
            "n_positives": int(pos.sum()),
            "exact_overlap_with_synthetic": leakage_check(df, a.synthetic_dir),
            "notes": [
                "url_defang is a representation probe, not a validated evasion.",
                "URL-only is invariant to lexical_proxy / neutral_hidden_html by construction.",
                "Rows with leakage_flag=True must NOT be reported as independent results.",
                f"std over seeds = training seeds only; {n} emails are not {n * n_runs} independent observations.",
            ],
        }
    )
    (OUT_PUBLIC / "exp42_provenance.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n=== AGGREGATE (subset = all messages) ===")
    show = agg[agg.subset == "all"][
        [
            "model",
            "leakage_flag",
            "condition",
            "f1_mean",
            "f1_std",
            "recall_mean",
            "fpr_mean",
            "asr_mean",
            "asr_den",
            "coverage_eligible",
        ]
    ]
    print(show.round(4).to_string(index=False))
    print("\nOverlap with synthetic splits:", extra["exact_overlap_with_synthetic"])
    print("URL score max |diff| (text-only attacks):", extra["url_score_max_abs_diff"])
    print("\nPublic aggregates :", OUT_PUBLIC)
    print("Private predictions:", OUT_PRIVATE / "exp42_predictions_private.csv")


if __name__ == "__main__":
    main()