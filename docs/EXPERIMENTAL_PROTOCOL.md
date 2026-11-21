# CYB-5 — Experimental protocol and comparison rules

**Version:** 2026-11-21. **Status:** exploratory experiments are documented; final benchmark protocol not yet fully executed.

## Research objective

Determine whether model-based email classifiers withstand evasion-style transformations compared with classical extracted-feature classifiers. Compare clean performance, attack-conditioned degradation, false positives on legitimate mail and inference cost.

## Classes and evidence provenance

| Source | Original label semantics | What it can support |
|---|---|---|
| CEAS-08 | Corpus spam/ham | Exploratory spam/ham training and carefully labeled candidate pool; **not** verified phishing-by-default. |
| Nazario | Corpus phishing-positive | Exploratory phishing-positive candidates; no intrinsic legitimate class for FPR. |
| SpaPhish | Corpus phishing/legitimate | Spanish-language external exploratory assessment already inspected; language/domain mismatch and no pristine holdout. |
| Phishing-validation v3 | Mainly generated/synthetic labels | Controlled template-separated training/reproducibility; synthetic high accuracy does not establish deployment readiness. |

Do not conflate original corpus labels with independent human phishing verification. Use an explicit `label_provenance` for every validation or test record.

### Experiment 37A: CEAS label-quality audit

- Drawn 80 messages, 40 from each original CEAS numeric class, **restricted to grouped train + validation**. Excludes existing CEAS test. The balanced sample is not a prevalence estimate.
- An initial *human two-class pass* produced 48 `ham` and 32 `spam`. A later separate-LLM suggestion proposed 8 `PHISHING` and 24 `SPAM` among the 32; these are **AI-generated proposals**.
- Eight proposed phishing messages received an additional AI assessment of `UNCERTAIN` because sender/target authenticity was not established. **These are not independent human final judgments.**
- Final allowed categories: `PHISHING`, `SPAM`, `LEGITIMATE`, `UNCERTAIN`. Two independent blind human reviews followed by disagreement adjudication are needed before claiming human-verified labels or inter-rater Cohen's kappa.
- Historical AI assistance, including email texts previously shared with chat services, must be disclosed. Do not imply all material stayed local.

### Experiment 38: fixed weak-label pipeline pilot

- Sampling: seed 42; 80 Nazario corpus positives, 80 CEAS corpus ham from original group train + validation only.
- All positive records are from Nazario and all negatives from CEAS. **Source/class confounding is complete.** Existing CEAS grouped test is excluded but previously inspected, so this is neither an independent test of CEAS nor a validated phishing benchmark.
- Threshold fixed at 0.50; not optimized on the pilot.
- Applies transformations to positives only. Negatives are unchanged and thus FPR does not vary by attack by construction.
- For details run `src/evaluation/experiment38_distilbert_pilot.py --help`.

### Experiment 39: exploratory threshold sweep

- Tested 0.50, 0.95, 0.98, 0.985, 0.99, 0.995 **on the already-inspected 160-email pilot**.
- Apparent best F1 among those tested: 0.812 at threshold 0.98, with FPR 0.263. This is **post-hoc analysis**, not an unbiased claim of calibrated or improved performance.
- **Never select a final threshold from this pilot and then report pilot results as independent evaluation.** A separate clean validation partition is required.

### Experiment 40: three independent fine-tunings, checkpoint reuse

- Seeds: 42, 123, 2026. Base: `distilbert/distilbert-base-uncased`.
- Training: 1307 synthetic-heavy messages; validation: 277; pre-inspected template-separated synthetic test: 316; original exploratory subset: 100.
- Configuration: max length 256, batch 16, learning rate 2e-5, up to three epochs, weight decay 0.01. Validation best-epoch rule uses F1 at threshold 0.50, with earliest exact tie. Post-checkpoint threshold selected on synthetic validation only with target validation FPR ≤ 0.05.
- In the uploaded results package, all three selected epochs = 1 and thresholds = 0.50; all three yielded synthetic test F1 = 1.0. The three completed models were **recovered/reused by the last notebook execution**, not re-fine-tuned there. Original per-epoch training histories are incomplete; do not claim complete three-seed loss curves.
- Test and original exploratory partitions have previously been inspected; not untouched final tests.

### Experiment 41: same 160-mail pilot across three checkpoints

- Reuses the **original Exp. 38 manifest** and the fixed Exp. 40 validation-selected threshold of 0.50 for each seed.
- Clean F1 across seeds: 0.6870, 0.6840, 0.7222; mean 0.6977, sample SD 0.0213.
- Clean recall: 0.9875, 0.9875, 0.9750; mean 0.9833.
- Clean FPR: 0.8875, 0.9000, 0.7250; mean 0.8375 (sample SD 0.0976).
- Mean F1: lexical proxy 0.6977 (42 changed positives); neutral hidden HTML 0.7054 (80 changed); URL defang 0.6977 (13 changed). No detected-to-missed flips observed.
- Absence of observed flips in these weak stress tests **is not proof of adversarial robustness**.

## Attack taxonomy and validity

| Name | Intended probe | Current pilot caveat |
|---|---|---|
| `lexical_proxy` | Deterministic conservative vocabulary substitutions | **Not** a certified semantic paraphrase. Only 42/80 corpus positives changed. |
| `neutral_hidden_html` | Append neutral comment-like text | Not an adversarial request; markup/tokenization sensitivity probe. 80/80 changed. |
| `url_defang` | Rewrite URL syntax to hxxp / `[.]` | Reduces clickability and can change phishing functionality; **not certified intent-preserving**. Only 13/80 positives had text-detectable URLs. |

Before a *final* evasion claim, establish a threat model, validate attack eligibility, confirm functional and malicious-intent preservation by reviewers where applicable, and report non-applicable transformations separately.

## Completed classical work and remaining comparison

TF-IDF + Logistic Regression on CEAS (random/grouped), TREC and synthetic templates, ComplementNB, MiniLM + LR, classical robustness tests and ablations are already implemented. Reuse `src/models/train_phishing_templates.py` and its matching private checkpoint for the common evaluation after checking exact and near-duplicate overlap. The historical validation-selected synthetic TF-IDF threshold is 0.44; any new threshold rule must be predeclared and selected using independent validation only. Do not rerun or replace historical experiments solely to generate higher scores. The missing implementations are URL-only and the generative LLM; the benchmark, attack validation and paired comparison remain shared tasks. See [corrected roadmap](CYB5_RESEARCH_ROADMAP.md) and [handoff](TEAM_HANDOFF.md).

## Common benchmark contract for all teams

**Do not reuse the CEAS training subset as an independent evaluation set for a baseline trained on CEAS.** A fair final comparison must be leakage-checked and source aware.

Each team receives one immutable, approved **private** dataset manifest with shared IDs and original texts. Keep a cryptographic SHA-256 of the local manifest, but do not publish private identifiers or message content. Each team must use:

1. The same sample IDs and reference labels, with class/label provenance.
2. Identical clean texts and byte-identical precomputed attack variants (or matching verified checksums); no independently improvised prompts/attacks for the common table.
3. The same training/validation/test partition boundaries and documented model training provenance.
4. Frozen parameters and thresholds selected without using evaluation labels. LLM prompt and decoding options frozen before testing.
5. A per-record private prediction table (kept local): `sample_id,condition,model,seed,y_true,y_pred,score,eligible,changed,latency_seconds`. Omit `score` if a generative model has no meaningful probability.
6. A sanitized public aggregate: `model,seed,condition,n,eligible_n,changed_n,tp,fp,tn,fn,precision,recall,f1,fpr,attack_successes,valid_attack_n,latency_mean_seconds,benchmark_status`.

Define phishing as positive class `1`; legitimate as negative `0` **only for records whose semantics have been validated**. `SPAM` and `UNCERTAIN` must not be silently relabeled into that binary target.

## Measures

- Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = 2·precision·recall / (precision+recall).
- FPR = FP / (FP + TN), computed on validated legitimate negatives, with denominator reported.
- Clean→miss evasion success = originally detected eligible positives changed to missed / originally detected eligible **validly attacked** positives. Never divide by all 80 when only 13 had URLs.
- Always report sample counts, invalid model outputs and coverage of successful text transformations.
- Record mean and sample SD across **independent training seeds** and, where meaningful, paired uncertainty estimates on identical evaluation messages (e.g., bootstrap by group). Three inference repeats of one checkpoint do not count as training seeds.
- Inference timing must specify CPU/GPU, batch size, model loading excluded or included, tokenizer included or excluded, warmup, and measurement conditions. Compare hardware costs cautiously; unknown cloud pricing is **not** zero-cost inference.

## Claims to avoid

- “CEAS label 1 = confirmed phishing.”
- “The 160-mail pilot is a representative or unseen real-world phishing test.”
- “100% synthetic test F1 proves real-world security.”
- “The model is adversarially robust because three simple modifications caused zero misses.”
- “The 80 review messages provide a human-human kappa” without two genuine independent human raters.
- “We chose 0.98 on validation” when it was first inspected on the Exp. 38 pilot.

**Milestones:** 24 November 2026 for experimental evidence; 8 December 2026 for 8–12-page two-column article, 12–15 slides, code, cited literature, limitations and AI-assistance declaration.
