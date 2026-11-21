#!/usr/bin/env python3
"""CYB-5 Exp38: local DistilBERT robustness pipeline shakedown.

Important: Nazario positive and CEAS ham are *corpus labels*, not independently
validated phishing/legitimate truth. Source-confounded exploratory results only.
This script never downloads model weights or publishes raw messages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
THRESHOLD = 0.50  # fixed, not selected on pilot sample


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_corpus(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, dtype={"text_hash": "string"})
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if df["text"].isna().any():
        df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


def split_hashes(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing grouped split file: {path}. "
            "Use the original CEAS grouped split from Experiment 37A."
        )
    x = pd.read_csv(path, dtype={"text_hash": "string"})
    if "text_hash" not in x:
        raise ValueError(f"{path} requires a text_hash column")
    if x["text_hash"].isna().any() or x["text_hash"].duplicated().any():
        raise ValueError(f"Missing or duplicate hashes in {path}")
    return set(x["text_hash"].astype(str))


def prepare(root: Path, out: Path, n: int, seed: int, force: bool) -> Path:
    manifest = out / "pilot_manifest.csv"
    if manifest.exists() and not force:
        print(f"Existing manifest kept: {manifest}")
        print("Use --force to intentionally replace it; otherwise run --mode run.")
        return manifest

    naz = read_corpus(root / "data/processed/nazario_external.csv", {"text", "label"})
    ceas = read_corpus(
        root / "data/processed/ceas_clean.csv", {"text", "label", "text_hash"}
    )
    ids_base = root / "results/metrics"
    train = split_hashes(ids_base / "group_train_ids.csv")
    val = split_hashes(ids_base / "group_validation_ids.csv")
    test = split_hashes(ids_base / "group_test_ids.csv")
    if (train & val) or (train & test) or (val & test):
        raise ValueError("CEAS group train/val/test hashes overlap")
    ceas_hash = set(ceas["text_hash"].astype(str))
    if ceas_hash != (train | val | test):
        raise ValueError("CEAS split hashes don't match the cleaned CEAS dataset")
    ceas["text_hash"] = ceas["text_hash"].astype(str)
    ceas_safe_pool = ceas[
        (ceas["label"] == 0) & ceas["text_hash"].isin(train | val)
    ].copy()
    naz_pos_pool = naz[naz["label"] == 1].copy()

    if min(len(ceas_safe_pool), len(naz_pos_pool)) < n:
        raise ValueError(
            f"Not enough pool rows for n={n}: Nazario={len(naz_pos_pool)}, "
            f"CEAS non-test ham={len(ceas_safe_pool)}"
        )
    selected_naz = naz_pos_pool.sample(n=n, random_state=seed).reset_index(drop=True)
    selected_ceas = ceas_safe_pool.sample(n=n, random_state=seed + 1).reset_index(drop=True)
    rows: list[dict] = []
    for i, row in selected_naz.iterrows():
        t = str(row["text"])
        rows.append({
            "sample_id": f"NAZ-{i + 1:03d}", "source": "Nazario",
            "partition": "external_explored", "label": 1,
            "label_provenance": "corpus_positive_not_manually_verified",
            "text_hash": str(row.get("text_hash", fingerprint(t))), "text": t,
        })
    for i, row in selected_ceas.iterrows():
        t = str(row["text"])
        h = str(row["text_hash"])
        rows.append({
            "sample_id": f"CEAS-{i + 1:03d}", "source": "CEAS_08",
            "partition": "train" if h in train else "validation", "label": 0,
            "label_provenance": "corpus_ham_not_manually_verified",
            "text_hash": h, "text": t,
        })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    if df["text_hash"].duplicated().any():
        raise ValueError("Repeated text hash in selected pilot; inspect source data")
    if set(df["text_hash"]) & test:
        # Some cross-corpus identical hash possible; exclude to be conservative.
        raise ValueError("Pilot overlaps CEAS existing grouped test partition")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    (out / "pilot_provenance.json").write_text(json.dumps({
        "task": "exploratory_classification_pipeline_shakedown_only",
        "ceas_labels": "legacy ham/spam, NOT verified phishing",
        "nazario_positive_labels": "corpus-provided phishing, not manually verified",
        "independent_unseen_test": False,
        "source_confounding": "all positives Nazario; all negatives CEAS",
        "seed_sampling_only": seed,
        "n_each_class": n,
        "original_ceas_test_excluded": True,
        "frozen_threshold": THRESHOLD,
        "manifest_contains_raw_messages": True,
    }, indent=2), encoding="utf-8")
    print(f"Wrote PRIVATE manifest ({len(df)} records): {manifest}")
    print("Positive with detectable text URL:", int(df[df.label == 1].text.str.contains(URL_RE).sum()))
    print("WARNING: weak labels and complete source/class confounding; exploratory only.")
    return manifest


def lexical_proxy(t: str) -> str:
    # Limited, fixed substitutions; NOT a validated semantic paraphrase.
    swaps = [
        (r"\bsign in\b", "log in"),
        (r"\bclick here\b", "select this link"),
        (r"\bverify\b", "confirm"),
        (r"\bimmediately\b", "promptly"),
        (r"\bnotification\b", "notice"),
    ]
    for pattern, repl in swaps:
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return t


def neutral_hidden_html(t: str) -> str:
    # Neutral local text that does not add, redirect or strengthen a lure.
    return t + "\n<!-- Email archive metadata: automated document index. -->"


def url_defang(t: str) -> str:
    # This deliberately makes URLs non-clickable; representation stress test,
    # not a validated intent-preserving phishing attack.
    def sub(match: re.Match[str]) -> str:
        s = match.group(0)
        # Keep trailing sentence punctuation as text, not part of the URL.
        url = s.rstrip(".,;:!?)")
        trailer = s[len(url):]
        url = re.sub(r"^https://", "hxxps://", url, flags=re.I)
        url = re.sub(r"^http://", "hxxp://", url, flags=re.I)
        return url.replace(".", "[.]") + trailer
    return URL_RE.sub(sub, t)


ATTACKS = {
    "lexical_proxy": lexical_proxy,
    "neutral_hidden_html": neutral_hidden_html,
    "url_defang": url_defang,
}


def load_checkpoint(model_dir: Path):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise RuntimeError("Install local dependencies: torch transformers") from e

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model checkpoint missing: {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local-only model {model_dir} on {device} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(device)
    except Exception as e:
        raise RuntimeError(
            f"Could not load local model/tokenizer from {model_dir}. "
            "Pass the correct saved checkpoint directory using --model-dirs."
        ) from e
    model.eval()
    mapping = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    if "phishing" in mapping:
        positive_id = mapping["phishing"]
    else:
        raise ValueError(f"Cannot find PHISHING output class in id2label: {mapping}")
    return model, tokenizer, device, positive_id


def predict_probs(checkpoint, texts: list[str], batch_size: int) -> tuple[np.ndarray, float]:
    import torch
    model, tokenizer, device, positive_id = checkpoint
    # Warm up the model once, then time tokenization + forward pass for each batch.
    if texts:
        warmup = tokenizer([texts[0]], truncation=True, padding=True, max_length=256,
                           return_tensors="pt")
        with torch.inference_mode():
            model(**{k: v.to(device) for k, v in warmup.items()})
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    probs: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i + batch_size], truncation=True,
                               padding=True, max_length=256, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            probs.extend(torch.softmax(logits, dim=1)[:, positive_id].cpu().tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    return np.asarray(probs, dtype=float), time.perf_counter() - started


def score(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run(root: Path, manifest: Path, models: list[Path], batch_size: int, seed: int) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(f"Pilot manifest not found: {manifest}; run --mode prepare")
    df = pd.read_csv(manifest, keep_default_na=False)
    for col in ["sample_id", "text", "label", "source", "label_provenance"]:
        if col not in df:
            raise ValueError(f"Pilot manifest missing {col}")
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    y = df.label.astype(int).to_numpy()
    if set(y) != {0, 1}:
        raise ValueError("Pilot requires both labels 0 and 1")
    if not all("not_manually_verified" in s for s in df.label_provenance):
        print("NOTE: Label-provenance field differs from the prepared weak-label pilot")
    texts = df.text.astype(str).tolist()
    out = root / "results/metrics/exp38_pilot"
    out.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    all_predictions: list[dict] = []
    pos = y == 1

    for model_i, model_dir in enumerate(models, start=1):
        model_dir = model_dir if model_dir.is_absolute() else (root / model_dir)
        model_name = model_dir.name
        print(f"\n--- CHECKPOINT {model_i}/{len(models)}: {model_dir} ---")
        checkpoint = load_checkpoint(model_dir)
        clean_prob, clean_seconds = predict_probs(checkpoint, texts, batch_size)
        clean_metrics = score(y, clean_prob)
        clean_pred = clean_prob >= THRESHOLD
        print(f"CLEAN F1 {clean_metrics['f1']:.4f}; recall {clean_metrics['recall']:.4f}; "
              f"FPR {clean_metrics['fpr']:.4f}")

        for attack_name, attack_fn in ATTACKS.items():
            eligible = np.asarray([
                bool(URL_RE.search(t)) if attack_name == "url_defang" else True
                for t in texts
            ], dtype=bool) & pos
            modified_texts = [
                attack_fn(t) if is_pos and is_eligible else t
                for t, is_pos, is_eligible in zip(texts, pos, eligible)
            ]
            changed = np.asarray([a != b for a, b in zip(texts, modified_texts)])
            # Keep eligibility honest: if no modification occurred, this isn't an attack.
            eligible &= changed
            att_prob, att_seconds = predict_probs(checkpoint, modified_texts, batch_size)
            attacked_metrics = score(y, att_prob)
            attacked_pred = att_prob >= THRESHOLD
            originally_correct = pos & eligible & clean_pred
            successful_flip = originally_correct & (~attacked_pred)
            count_origin_correct = int(originally_correct.sum())
            metric_row = {
                "model_checkpoint": model_name,
                "checkpoint_path": str(model_dir),
                "attack": attack_name,
                "dataset_status": "EXPLORATORY_WEAK_CORPUS_LABELS_SOURCE_CONFOUNDED",
                "model_runs_supplied": len(models),
                "sampling_seed": seed,
                "threshold_frozen": THRESHOLD,
                "n_positive": int(pos.sum()),
                "n_negative": int((~pos).sum()),
                "eligible_positive_messages": int(eligible.sum()),
                "changed_positive_messages": int((pos & changed).sum()),
                "initially_detected_eligible_positives": count_origin_correct,
                "detected_to_missed_flips": int(successful_flip.sum()),
                "detected_to_missed_rate": (
                    float(successful_flip.sum() / count_origin_correct)
                    if count_origin_correct else None
                ),
                "clean_seconds_per_message": clean_seconds / len(texts),
                "attacked_seconds_per_message": att_seconds / len(texts),
                **{f"clean_{k}": v for k, v in clean_metrics.items()},
                **{f"attacked_{k}": v for k, v in attacked_metrics.items()},
                "f1_delta": attacked_metrics["f1"] - clean_metrics["f1"],
                "attack_intent_preserving_certified": False,
            }
            all_metrics.append(metric_row)
            for i, record in df.iterrows():
                all_predictions.append({
                    "checkpoint": model_name, "sample_id": record.sample_id,
                    "source": record.source, "attack": attack_name, "label": int(y[i]),
                    "positive_eligible": bool(eligible[i]),
                    "text_changed": bool(changed[i]),
                    "clean_probability": round(float(clean_prob[i]), 7),
                    "attack_probability": round(float(att_prob[i]), 7),
                    "clean_pred": int(clean_pred[i]),
                    "attacked_pred": int(attacked_pred[i]),
                })
            print(f"{attack_name:<21} eligible={int(eligible.sum()):>3} "
                  f"F1={attacked_metrics['f1']:.4f} "
                  f"clean->miss={int(successful_flip.sum())}/{count_origin_correct}")

    if len(models) < 3:
        print("WARNING: fewer than 3 independently trained checkpoints supplied. "
              "This DOES NOT fulfill the 3-training-seed research requirement.")

    pd.DataFrame(all_metrics).to_csv(out / "exp38_weak_label_metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(out / "exp38_predictions_NO_RAW_TEXT.csv", index=False)
    (out / "exp38_weak_label_metrics.json").write_text(
        json.dumps({
            "status": "exploratory_only_not_verified_phishing_performance",
            "warning": "label/source confounding; no manual verification, "
                       "no independently untouched CEAS test",
            "seed_requirement_fulfilled": False,  # training-seed provenance not known
            "independent_training_seeds_must_be_verified_externally": True,
            "threshold": THRESHOLD,
            "positive_only_stress_tests": True,
            "negative_fpr_invariant_by_design": True,
            "url_defang_not_intent_preserving": True,
            "attacks": all_metrics,
        }, indent=2), encoding="utf-8"
    )
    print("\nWrote result files to:", out)
    print("NO raw message texts written into results/. No trained model changed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["prepare", "run", "all"], default="all")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--n-per-class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model-dirs", nargs="+", type=Path, default=[Path("models/distilbert_cyb5")])
    ap.add_argument("--force", action="store_true", help="Intentionally regenerate private manifest")
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = root / "data/audit/cyb5_experiment38_pilot/pilot_manifest.csv"
    if args.n_per_class < 1:
        ap.error("--n-per-class must be positive")
    if args.mode in ("prepare", "all"):
        prepare(root, manifest.parent, args.n_per_class, args.seed, args.force)
    if args.mode in ("run", "all"):
        run(root, manifest, args.model_dirs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
