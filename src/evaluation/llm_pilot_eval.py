#!/usr/bin/env python3
"""CYB-5 LLM pilot: zero-shot local LLM on the SAME 160 messages and 4 conditions.

Exploratory only. Nazario positives and CEAS negatives are *corpus labels* and are
perfectly confounded with the source; nothing here is a validated estimate of
real-world phishing performance, and none of the attacks is certified to
preserve attack intent.

Transformations are imported from ``experiment38_distilbert_pilot`` (identical to
the ones copied into the Experiment 41 notebook), never re-implemented.

Modes (run from the repository root):
    python -m src.evaluation.llm_pilot_eval --mode validate   # synthetic, non-pilot
    python -m src.evaluation.llm_pilot_eval --mode run        # 160 x 4 conditions

Private outputs (git-ignored, under data/): per-message predictions with IDs, no text.
Public candidates (review before commit): results/metrics/llm_aggregate.csv and
results/metrics/llm_run_record.json -- aggregates only, no text, IDs, URLs or paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation import experiment38_distilbert_pilot as exp38
from src.models.llm_detector import LLMConfig, LLMDetector, to_pred

CONDITIONS = ["clean", "lexical_proxy", "neutral_hidden_html", "url_defang"]
BENCHMARK_STATUS = "EXPLORATORY_SOURCE_CLASS_CONFOUNDED"
# Number of positives actually changed in Exp. 41 (public summary); used to prove
# that the supplied manifest is the original pilot and not a fresh sample.
EXPECTED_CHANGED = {"lexical_proxy": 42, "neutral_hidden_html": 80, "url_defang": 13}
MANIFEST_REL = Path("data/audit/cyb5_experiment38_pilot/pilot_manifest.csv")
PRIVATE_DIR_REL = Path("data/audit/cyb5_llm_pilot")
PUBLIC_DIR_REL = Path("results/metrics")
VALIDATION_REL = Path("configs/llm_validation_synthetic.jsonl")

PRED_COLUMNS = [
    "model", "model_version", "seed", "sample_id", "condition", "source", "label", "pred",
    "score", "eligible", "text_changed", "latency_ms",
    "llm_label", "output_class", "refusal", "context_overflow", "capped",
    "prompt_tokens", "gen_tokens", "raw_chars", "error",
]


# --------------------------------------------------------------------------- data
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: Path, expected_n: int = 160) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Private pilot manifest not found: {path}\n"
            "Use the ORIGINAL Experiment 38 file (data/audit/cyb5_experiment38_pilot/"
            "pilot_manifest.csv); do not regenerate it."
        )
    df = pd.read_csv(path, keep_default_na=False)
    required = {"sample_id", "source", "label", "label_provenance", "text"}
    if missing := required - set(df.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if len(df) != expected_n or df.sample_id.duplicated().any():
        raise ValueError(f"Expected exactly {expected_n} unique sample_id, got {len(df)}")
    if set(df.label.astype(int)) != {0, 1} or (df.text.astype(str).str.strip() == "").any():
        raise ValueError("Manifest needs both labels and non-empty texts")
    df["label"] = df.label.astype(int)
    if df.label.value_counts().to_dict() != {0: expected_n // 2, 1: expected_n // 2}:
        raise ValueError("Manifest must be balanced (80 positives / 80 negatives)")
    return df.reset_index(drop=True)


def build_variants(texts: list[str], pos: np.ndarray):
    """Exactly the Experiment 41 logic: positives only, eligible = actually changed."""
    variants = {"clean": list(texts)}
    changed = {"clean": np.zeros(len(texts), dtype=bool)}
    eligible = {"clean": np.zeros(len(texts), dtype=bool)}
    for name, fn in exp38.ATTACKS.items():
        proposed = [
            fn(t) if p and (name != "url_defang" or exp38.URL_RE.search(t)) else t
            for t, p in zip(texts, pos)
        ]
        ch = np.array([a != b for a, b in zip(texts, proposed)], dtype=bool)
        variants[name], changed[name], eligible[name] = proposed, ch, pos & ch
    return variants, changed, eligible


def attack_code_sha256() -> str:
    src = "".join(inspect.getsource(f) for f in exp38.ATTACKS.values()) + exp38.URL_RE.pattern
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- metrics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def confusion(y: np.ndarray, p: np.ndarray) -> tuple[int, int, int, int]:
    return (int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 0) & (p == 0)).sum()), int(((y == 1) & (p == 0)).sum()))


def prf(tp: int, fp: int, tn: int, fn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / max(tp + fp + tn + fn, 1), "precision": prec,
            "recall": rec, "f1": f1, "fpr": fp / (fp + tn) if fp + tn else float("nan")}


def aggregate(preds: pd.DataFrame, model_version: str, invalid_policy: str) -> pd.DataFrame:
    """One anonymised row per condition. Needs per-message predictions of all conditions."""
    alt = "phishing" if invalid_policy == "legitimate" else "legitimate"
    base = preds[preds.condition == "clean"].set_index("sample_id")
    ids = list(base.index)
    y = base.loc[ids, "label"].to_numpy(int)
    clean_pred = base.loc[ids, "pred"].to_numpy(int)
    clean_f1 = prf(*confusion(y, clean_pred))["f1"]
    rows = []
    for cond in CONDITIONS:
        d = preds[preds.condition == cond].set_index("sample_id").loc[ids]
        pred = d["pred"].to_numpy(int)
        elig = d["eligible"].to_numpy(bool)
        chg = d["text_changed"].to_numpy(bool)
        pos = y == 1
        tp, fp, tn, fn = confusion(y, pred)
        m = prf(tp, fp, tn, fn)
        labels = d["llm_label"].fillna("").to_numpy(str)
        alt_pred = np.array([to_pred(l if l else None, alt) for l in labels])
        m_alt = prf(*confusion(y, alt_pred))
        denom = pos & elig & (clean_pred == 1)
        flips = denom & (pred == 0)
        rec_lo, rec_hi = wilson(tp, tp + fn)
        fpr_lo, fpr_hi = wilson(fp, fp + tn)
        lat = d["latency_ms"].astype(float).to_numpy() / 1000.0
        ok = d["error"].fillna("").to_numpy(str) == ""
        klass = d["output_class"].to_numpy(str)
        rows.append({
            "model": "LLM_zero_shot", "model_version": model_version, "seed": "not_applicable",
            "condition": cond, "benchmark_status": BENCHMARK_STATUS,
            "n": len(ids), "n_positive": int(pos.sum()), "n_negative": int((~pos).sum()),
            "eligible_positive": int(elig.sum()), "changed_positive": int((pos & chg).sum()),
            "coverage_eligible": float(elig.sum() / pos.sum()),
            "coverage_changed": float((pos & chg).sum() / pos.sum()),
            "asr_denominator": int(denom.sum()), "detected_to_missed_flips": int(flips.sum()),
            "asr": float(flips.sum() / denom.sum()) if denom.sum() else float("nan"),
            "missed_to_detected_flips": int((pos & elig & (clean_pred == 0) & (pred == 1)).sum()),
            "unchanged_text_pred_diffs": int(((~chg) & (pred != clean_pred)).sum()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, **m,
            "delta_f1_vs_clean": m["f1"] - clean_f1,
            "recall_ci95_low": rec_lo, "recall_ci95_high": rec_hi,
            "fpr_ci95_low": fpr_lo, "fpr_ci95_high": fpr_hi,
            "invalid_policy": invalid_policy,
            "invalid_n": int((klass == "invalid").sum()), "error_n": int((klass == "error").sum()),
            "invalid_rate": float(((klass == "invalid") | (klass == "error")).mean()),
            "lenient_parse_n": int((klass == "lenient").sum()),
            "refusal_n": int(d["refusal"].astype(bool).sum()),
            "context_overflow_n": int(d["context_overflow"].astype(bool).sum()),
            "capped_n": int(d["capped"].astype(bool).sum()),
            "f1_alt_invalid_policy": m_alt["f1"], "fpr_alt_invalid_policy": m_alt["fpr"],
            "latency_mean_s": float(np.mean(lat[ok])) if ok.any() else float("nan"),
            "latency_median_s": float(np.median(lat[ok])) if ok.any() else float("nan"),
            "latency_p95_s": float(np.percentile(lat[ok], 95)) if ok.any() else float("nan"),
            "prompt_tokens_mean": float(pd.to_numeric(d["prompt_tokens"], errors="coerce").mean()),
            "gen_tokens_mean": float(pd.to_numeric(d["gen_tokens"], errors="coerce").mean()),
            "valid_attack_certified": False,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- running
def _read_done(pred_path: Path) -> set[tuple[str, str]]:
    if not pred_path.is_file():
        return set()
    old = pd.read_csv(pred_path, keep_default_na=False, dtype=str)
    return set(zip(old.sample_id, old.condition))


def run_conditions(df: pd.DataFrame, variants, changed, eligible, detector, model_version: str,
                   pred_path: Path, progress_every: int = 20) -> pd.DataFrame:
    """Classify every message under every condition; append row by row (resumable)."""
    done = _read_done(pred_path)
    new_file = not pred_path.is_file()
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(df) * len(CONDITIONS)
    with open(pred_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PRED_COLUMNS)
        if new_file:
            writer.writeheader()
        count = len(done)
        for cond in CONDITIONS:
            for i, row in df.iterrows():
                if (row.sample_id, cond) in done:
                    continue
                p = detector.classify(variants[cond][i])
                writer.writerow({
                    "model": "LLM_zero_shot", "model_version": model_version,
                    "seed": "not_applicable", "sample_id": row.sample_id, "condition": cond,
                    "source": row.source, "label": int(row.label), "pred": p.pred, "score": "",
                    "eligible": str(bool(eligible[cond][i])).lower(),
                    "text_changed": str(bool(changed[cond][i])).lower(),
                    "latency_ms": round(p.latency_s * 1000, 2),
                    "llm_label": p.label or "", "output_class": p.output_class,
                    "refusal": p.refusal, "context_overflow": p.context_overflow,
                    "capped": p.capped, "prompt_tokens": p.prompt_tokens,
                    "gen_tokens": p.gen_tokens, "raw_chars": p.raw_chars, "error": p.error,
                })
                fh.flush()
                count += 1
                if count % progress_every == 0 or count == total:
                    print(f"  {count}/{total} predictions", flush=True)
    out = pd.read_csv(pred_path, keep_default_na=False)
    for col in ("refusal", "context_overflow", "capped"):
        out[col] = out[col].astype(str).str.lower().eq("true")
    for col in ("eligible", "text_changed"):
        out[col] = out[col].astype(str).str.lower().eq("true")
    return out


def hardware_record() -> dict:
    gpu = "none detected"
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip() or gpu
        except Exception:  # pragma: no cover - environment dependent
            gpu = "nvidia-smi failed"
    return {"gpu": gpu, "platform": platform.platform(), "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__, "batch_size": 1}


def length_summary(texts: Iterable[str]) -> dict:
    n = np.array([len(t) for t in texts])
    return {"chars_median": int(np.median(n)), "chars_p95": int(np.percentile(n, 95)),
            "chars_max": int(n.max())}


# --------------------------------------------------------------------------- modes
def mode_validate(root: Path, cfg: LLMConfig, det: LLMDetector) -> None:
    """Smoke test on SYNTHETIC inert messages: parsing, invalid rate, injection."""
    rows = [json.loads(l) for l in (root / VALIDATION_REL).read_text(encoding="utf-8").splitlines() if l.strip()]
    det.warm_up()
    res = [(r, det.classify(r["text"])) for r in rows]
    plain = [(r, p) for r, p in res if r["kind"] == "plain"]
    inj = [(r, p) for r, p in res if r["kind"] == "injection"]
    classes = pd.Series([p.output_class for _, p in res]).value_counts().to_dict()
    report = {
        "validation_set": "synthetic, inert, disjoint from the pilot",
        "n": len(res), "output_classes": classes,
        "invalid_or_error_rate": float(np.mean([p.output_class in ("invalid", "error") for _, p in res])),
        "refusals": int(sum(p.refusal for _, p in res)),
        "plain_accuracy_indicative": float(np.mean([p.pred == r["label"] for r, p in plain])),
        "injection_cases_correct": f"{sum(p.pred == r['label'] for r, p in inj)}/{len(inj)}",
        "prompt_sha256": det.prompt_sha256, "invalid_policy": cfg.invalid_policy,
    }
    out = root / PRIVATE_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("If the prompt needs a change, create llm_prompt_v2.txt + a new config (v1 stays frozen).")


def mode_run(root: Path, cfg: LLMConfig, det: LLMDetector, manifest: Path,
             restart: bool, allow_other_manifest: bool) -> None:
    try:
        info = det.describe_model()
    except Exception as exc:
        raise SystemExit(f"Ollama is not reachable at {cfg.ollama_url} ({type(exc).__name__}). "
                         "See docs/LLM_OLLAMA_SETUP.md") from exc
    model_version = f"{cfg.model}@{str(info['digest'])[:12]}"
    df = load_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    texts = df.text.astype(str).tolist()
    pos = df.label.to_numpy(int) == 1
    variants, changed, eligible = build_variants(texts, pos)

    observed = {c: int((pos & changed[c]).sum()) for c in EXPECTED_CHANGED}
    print("Changed positives per attack:", observed, "| Exp.41 reference:", EXPECTED_CHANGED)
    if observed != EXPECTED_CHANGED and not allow_other_manifest:
        raise SystemExit("Manifest does not reproduce the Exp.41 coverage (42/80/13): "
                         "this is not the original pilot. Use --allow-other-manifest only knowingly.")
    print("Message length (characters, clean):", length_summary(texts))

    priv = root / PRIVATE_DIR_REL
    priv.mkdir(parents=True, exist_ok=True)
    pred_path, state_path = priv / "llm_predictions_PRIVATE.csv", priv / "llm_run_state.json"
    state = {"prompt_sha256": det.prompt_sha256, "model_digest": info["digest"],
             "manifest_sha256": manifest_sha, "model": cfg.model, "invalid_policy": cfg.invalid_policy}
    if restart:
        pred_path.unlink(missing_ok=True)
    elif pred_path.is_file():
        old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if old != state:
            raise SystemExit("Existing partial run used another prompt/model/manifest. "
                             "Use --restart to discard it.")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    warm = det.warm_up()
    print(f"Model loaded in {warm:.1f}s (excluded from latency). Running {len(df) * 4} predictions...")
    t0 = time.perf_counter()
    preds = run_conditions(df, variants, changed, eligible, det, model_version, pred_path)
    wall = time.perf_counter() - t0
    agg = aggregate(preds, model_version, cfg.invalid_policy)

    pub = root / PUBLIC_DIR_REL
    pub.mkdir(parents=True, exist_ok=True)
    agg.round(6).to_csv(pub / "llm_aggregate.csv", index=False)
    validation = priv / "llm_validation_report.json"
    record = {
        "experiment": "LLM zero-shot pilot", "benchmark_status": BENCHMARK_STATUS,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": cfg.model, "model_digest": info["digest"], "model_details": info["details"],
        "model_license_note": cfg.model_license_note, "ollama_version": info["ollama_version"],
        "config": {k: getattr(cfg, k) for k in ("temperature", "seed", "num_ctx", "num_predict",
                  "constrained_decoding", "invalid_policy", "max_email_chars", "timeout_s")},
        "prompt_file": cfg.prompt_file, "prompt_sha256": det.prompt_sha256,
        "manifest_sha256_private_file": manifest_sha, "attack_code_sha256": attack_code_sha256(),
        "changed_positives": observed, "conditions": CONDITIONS, "n_messages": len(df),
        "model_load_seconds_excluded": round(warm, 2), "wall_seconds": round(wall, 1),
        "hardware": hardware_record(),
        "validation_report": json.loads(validation.read_text(encoding="utf-8")) if validation.is_file() else None,
        "attacks_certified_intent_preserving": False,
    }
    (pub / "llm_run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    show = ["condition", "eligible_positive", "tp", "fp", "tn", "fn", "precision", "recall", "f1",
            "fpr", "delta_f1_vs_clean", "asr_denominator", "detected_to_missed_flips",
            "invalid_rate", "latency_mean_s"]
    print(agg[show].round(4).to_string(index=False))
    print(f"\nPublic candidates (review first): {pub/'llm_aggregate.csv'}, {pub/'llm_run_record.json'}")
    print(f"PRIVATE (IDs, git-ignored): {pred_path}")
    print("EXPLORATORY, source-confounded; not a real-phishing evaluation.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["validate", "run"], required=True)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--config", type=Path, default=Path("configs/llm_config_v1.json"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--restart", action="store_true", help="discard a partial private run")
    ap.add_argument("--allow-other-manifest", action="store_true")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    cfg = LLMConfig.load(root / args.config)
    det = LLMDetector(cfg, root)
    if args.mode == "validate":
        mode_validate(root, cfg, det)
    else:
        mode_run(root, cfg, det, args.manifest or root / MANIFEST_REL,
                 args.restart, args.allow_other_manifest)


if __name__ == "__main__":
    main()
