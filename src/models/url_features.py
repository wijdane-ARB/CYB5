"""Offline URL feature extraction + shared helpers (CYB-5 Exp42, task 2).
Never opens a link: pure string parsing only."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

# Same pattern as experiment38_distilbert_pilot.py (so "has a URL" means the same thing everywhere).
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
TRAILING = ".,;:!?)]}"
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc",
}
RISKY_TLDS = {
    "zip", "xyz", "top", "tk", "ml", "ga", "cf", "gq",
    "click", "work", "support", "loan", "icu",
}  # heuristic, documented
SPECIAL = set("@?&=%_~#")

_DEFANG_PATTERNS = [
    (re.compile(r"hxxps?", re.IGNORECASE), lambda m: m.group(0).replace("xx", "tt").replace("XX", "TT")),
    (re.compile(r"\[\.\]"), "."),
    (re.compile(r"\(\.\)"), "."),
    (re.compile(r"\{\.\}"), "."),
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"\[@\]"), "@"),
]

# Families are used for the ablation (remove one family at a time).
FEATURE_FAMILIES = {
    "count": ["has_url", "n_urls", "n_unique_hosts"],
    "length": ["url_len_max", "url_len_mean", "host_len_max", "path_len_max", "query_len_max"],
    "host": [
        "n_dots_max", "n_subdomains_max", "n_hyphens_host_max",
        "host_entropy_max", "longest_label_max",
    ],
    "chars": ["n_digits_max", "digit_ratio_max", "n_special_max", "n_slashes_max", "n_pct_max"],
    "flags": [
        "any_ip", "any_at", "any_punycode", "any_port", "any_shortener",
        "any_risky_tld", "any_plain_http", "frac_https",
    ],
}
FEATURE_NAMES = [n for names in FEATURE_FAMILIES.values() for n in names]
TEXT_CANDIDATES = ["text", "body", "email_text", "message", "content", "email"]
LABEL_CANDIDATES = ["label", "class", "target", "is_phishing"]


def reverse_defang(text: str) -> str:
    """Reconvertit les URLs defangées (hxxp://evil[.]com) en forme canonique (http://evil.com)."""
    if not isinstance(text, str):
        return text
    out = text
    for pattern, repl in _DEFANG_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def extract_urls(text) -> list[str]:
    """Extrait les URLs après reverse defanging."""
    normalized = reverse_defang("" if text is None else str(text))
    return [m.group(0).rstrip(TRAILING) for m in URL_RE.finditer(normalized)]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _one(url: str) -> dict:
    cand = "http://" + url if url.lower().startswith("www.") else url
    host = path = query = ""
    port = None
    try:
        s = urlsplit(cand)
        host = (s.hostname or "").lower()
        path, query = s.path, s.query
        try:
            port = s.port
        except ValueError:
            port = None
    except ValueError:
        pass
    labels = [x for x in host.split(".") if x]
    is_ip = bool(IP_RE.match(host)) or ":" in host
    digits = sum(c.isdigit() for c in url)
    low = url.lower()
    return {
        "len": len(url),
        "host": host,
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "dots": host.count("."),
        "subs": 0 if is_ip else max(len(labels) - 2, 0),
        "hyph_host": host.count("-"),
        "ent": _entropy(host),
        "longest": max((len(x) for x in labels), default=0),
        "digits": digits,
        "digit_ratio": digits / max(len(url), 1),
        "special": sum(c in SPECIAL for c in url),
        "slashes": url.count("/"),
        "pct": url.count("%"),
        "ip": is_ip,
        "at": "@" in url,
        "puny": "xn--" in host,
        "port": port is not None,
        "short": host in SHORTENERS or host.removeprefix("www.") in SHORTENERS,
        "risky": bool(labels) and labels[-1] in RISKY_TLDS,
        "plain": low.startswith("http://"),
        "https": low.startswith("https://"),
    }


def message_features(text) -> tuple[list[float], int]:
    """Message-level vector. No URL -> all zeros (has_url=0 is the explicit 'missing' flag)."""
    urls = extract_urls(text)
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    if urls:
        r = [_one(u) for u in urls]
        mx = lambda k: float(max(x[k] for x in r))
        f.update(
            has_url=1.0,
            n_urls=float(len(urls)),
            n_unique_hosts=float(len({x["host"] for x in r})),
            url_len_max=mx("len"),
            url_len_mean=float(np.mean([x["len"] for x in r])),
            host_len_max=mx("host_len"),
            path_len_max=mx("path_len"),
            query_len_max=mx("query_len"),
            n_dots_max=mx("dots"),
            n_subdomains_max=mx("subs"),
            n_hyphens_host_max=mx("hyph_host"),
            host_entropy_max=mx("ent"),
            longest_label_max=mx("longest"),
            n_digits_max=mx("digits"),
            digit_ratio_max=mx("digit_ratio"),
            n_special_max=mx("special"),
            n_slashes_max=mx("slashes"),
            n_pct_max=mx("pct"),
            any_ip=mx("ip"),
            any_at=mx("at"),
            any_punycode=mx("puny"),
            any_port=mx("port"),
            any_shortener=mx("short"),
            any_risky_tld=mx("risky"),
            any_plain_http=mx("plain"),
            frac_https=float(np.mean([x["https"] for x in r])),
        )
    return [f[n] for n in FEATURE_NAMES], len(urls)


def featurize(texts) -> np.ndarray:
    return np.asarray([message_features(t)[0] for t in texts], dtype=float).reshape(-1, len(FEATURE_NAMES))


# ---------------- shared helpers ----------------
def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def guess_column(df, explicit, candidates, what):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{what} column '{explicit}' not in {df.columns.tolist()}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot guess {what} column; columns = {df.columns.tolist()}")


def normalize_labels(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        y = s.astype(int)
    else:
        m = {
            "phishing": 1, "spam": 1, "malicious": 1, "1": 1, "true": 1,
            "legitimate": 0, "legit": 0, "ham": 0, "benign": 0, "safe": 0, "0": 0, "false": 0,
        }
        low = s.astype(str).str.strip().str.lower()
        unknown = sorted(set(low) - set(m))
        if unknown:
            raise ValueError(f"Unknown label values: {unknown}")
        y = low.map(m).astype(int)
    if not set(y.unique()) <= {0, 1}:
        raise ValueError("Labels must be binary 0/1")
    return y.to_numpy()


def norm_hash(text) -> str:
    return hashlib.sha256(" ".join(str(text).lower().split()).encode("utf-8")).hexdigest()


def binary_metrics(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not math.isnan(rec) else 0.0
    return {
        "n": int(len(y)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d