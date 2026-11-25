"""URL coverage by class (and source). Prints/saves counts only, never URLs."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
from url_features import (extract_urls, read_table, guess_column, normalize_labels,
                          TEXT_CANDIDATES, LABEL_CANDIDATES)

METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--text-col"); ap.add_argument("--label-col"); ap.add_argument("--group-col")
    a = ap.parse_args()
    for path in a.input:
        df = read_table(Path(path))
        t = guess_column(df, a.text_col, TEXT_CANDIDATES, "text")
        l = guess_column(df, a.label_col, LABEL_CANDIDATES, "label")
        df = df.assign(text=df[t].astype(str), label=normalize_labels(df[l]))
        df["url_count"] = df["text"].map(lambda x: len(extract_urls(x)))
        df["has_url"] = df["url_count"] > 0
        keys = ["label"] + ([a.group_col] if a.group_col and a.group_col in df else [])
        tbl = (df.groupby(keys).agg(messages=("has_url", "size"), with_url=("has_url", "sum"),
                                    total_urls=("url_count", "sum")).reset_index())
        tbl["coverage"] = tbl["with_url"] / tbl["messages"]
        print(f"\n=== {Path(path).name} : {len(df)} messages, "
              f"{int(df.has_url.sum())} with URL ({df.has_url.mean():.1%}) ===")
        print(tbl.round(4).to_string(index=False))
        if (tbl["coverage"] < 0.05).any():
            print("WARNING: at least one class/source has < 5% URL coverage.")
        out = METRICS / f"url_coverage_{Path(path).stem}.csv"
        tbl.to_csv(out, index=False)
        print("Saved:", out)


if __name__ == "__main__":
    main()