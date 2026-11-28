# Public result summaries

Only **sanitized, aggregate** results should be added to this directory in future commits. Existing tracked files must still be audited separately; `.gitignore` does not untrack them.

| File | Provenance | Limitations |
|---|---|---|
| `exp38_pilot_summary_public.csv` | Reported local Experiment 38 run, one checkpoint, 160 messages | Weak labels, perfectly source-confounded, attack validity not certified. |
| `exp40_three_seed_summary_public.csv` | Original submitted `CYB5_Experiment40_3Seed_Results.zip` | Synthetic-heavy; old test inspected; historical training logs incomplete. |
| `exp41_three_seed_summary_public.csv` | Original executed Exp. 41 notebook logs (three checkpoints) | Same previously inspected 160-message pilot; positive-only modifications. |
| `llm_aggregate.csv` | Local zero-shot LLM (`src/evaluation/llm_pilot_eval.py`), 160 messages x 4 conditions, one fixed configuration (no training seeds) | Same exploratory, source-confounded pilot; attacks not certified intent-preserving; latency is hardware-specific. Companion: `llm_run_record.json` (model digest, prompt hash, parameters). |

The public CSVs include **no raw message text, sample IDs, hashes, URLs, model weight paths, email addresses or private annotation decisions**. F1 is the positive-class F1; FPR denominator is the reported negative-class count. See `docs/EXPERIMENTAL_PROTOCOL.md` for definitions and context.

These files are not substitutes for the **private original experimental archives**, which should be preserved separately with appropriate access controls.
