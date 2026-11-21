#!/usr/bin/env python3
"""Private local-only source-evidence matching for disputed CEAS records.

Input is an ignored file in data/audit/ containing columns `review_id,snippet`.
No real email snippets are embedded in public repository code.
Does not authenticate sender/DKIM/SPF or assess maliciousness.
"""
import argparse
import csv
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd())
    ap.add_argument('--clues', type=Path, default=Path('data/audit/ceas_stageA/private/provenance_clues.csv'))
    args = ap.parse_args()
    root = args.root.resolve()
    clues = args.clues if args.clues.is_absolute() else root / args.clues
    src = root / 'data/raw/CEAS_08.csv'
    dst = root / 'data/audit/ceas_stageA/private/ceas_raw_evidence.csv'
    if not src.is_file() or not clues.is_file():
        raise FileNotFoundError(
            'Missing original CEAS input or private review clues. '
            f'Expected: {src} and {clues}'
        )
    import pandas as pd
    clue_df = pd.read_csv(clues, dtype=str, keep_default_na=False)
    if not {'review_id', 'snippet'}.issubset(clue_df):
        raise ValueError('Private clues CSV must have review_id,snippet columns')
    if clue_df.empty or clue_df.review_id.duplicated().any() or clue_df.snippet.eq('').any():
        raise ValueError('Clue list must be nonempty with unique IDs and nonempty snippets')
    df = pd.read_csv(src, engine='python')
    required = {'sender', 'subject', 'body', 'urls', 'label'}
    if missing := required - set(df):
        raise ValueError(f'Missing original CEAS fields: {sorted(missing)}')
    hay = df.subject.fillna('').astype(str) + '\n' + df.body.fillna('').astype(str)
    out = []
    for clue in clue_df.itertuples(index=False):
        hits = df[hay.str.contains(clue.snippet, case=False, regex=False, na=False)]
        row = {'review_id': clue.review_id, 'raw_hits': len(hits),
               'sender_display_only': '', 'subject': '', 'original_label': '',
               'raw_binary_urls_flag': '', 'visible_url_count': '',
               'interpretation': 'unmatched_or_ambiguous'}
        if len(hits) == 1:
            hit = hits.iloc[0]
            row.update(sender_display_only=str(hit.sender), subject=str(hit.subject),
                       original_label=str(hit.label), raw_binary_urls_flag=str(hit.urls),
                       visible_url_count=len(re.findall(r'(?i)\b(?:https?://|www\.)[^\s<>"\x27]+', str(hit.body))),
                       interpretation='sender_unverified_url_flag_not_proof_of_phishing')
        out.append(row)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys())
        writer.writeheader()
        writer.writerows(out)
    print('Private evidence saved:', dst)
    for row in out:
        print(row['review_id'], 'matches:', row['raw_hits'])
    print('WARNING: multiple matches or display-sender fields do not authenticate a message.')

if __name__ == '__main__':
    main()
