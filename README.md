# CYB-5 — Robustesse adversariale de la détection d'hameçonnage

**ENSET Mohammedia · Master Systèmes Distribués et Intelligence Artificielle · Projet d'IA avancée**

> **Research question:** Can a language-model-based phishing detector withstand evasion-oriented rewrites compared with classical feature-based detectors?
>
> **Question de recherche :** Un détecteur d'hameçonnage fondé sur un modèle de langage résiste-t-il à des messages réécrits pour l'évasion, comparé à un modèle classique sur variables extraites ?

**Status (21 November 2026):** TF-IDF + Logistic Regression (CEAS, grouped CEAS, TREC and template-separated synthetic data), ComplementNB, MiniLM + LR, DistilBERT and the associated ablation, robustness and error analyses have already been implemented and evaluated. Experiments 37A–41 are documented. The shared, independently validated final benchmark, **reuse of the existing TF-IDF implementation on that benchmark**, the new URL-feature classifier, the generative LLM comparison and final paired analysis **remain incomplete**. Do not repeat the completed historical experiments. This repository is a **research work in progress, not a deployable phishing filter**.

## What has been studied

| Component | Status | Notes |
|---|---|---|
| CEAS-08 audit | Preliminary | The source labels are spam/ham, **not independently verified phishing**; first-pass human labels, AI-assisted revisions and still-unresolved adjudications are kept distinct. |
| TF-IDF + Logistic Regression | Completed on historical corpora | See `src/models/train_baseline.py`, `train_group_baseline.py`, `train_meajor_baseline.py` and `train_phishing_templates.py`. Reuse the existing implementation for the leakage-controlled common comparison; no new TF-IDF implementation is required. |
| ComplementNB and MiniLM + LR | Completed historical comparisons | TREC calibration and paired bootstrap; frozen MiniLM embeddings on the synthetic template benchmark. |
| Classical robustness and ablations | Completed historical experiments | Neutral footer, case, punctuation, spacing, joint masking, augmentation, duplicate and threshold analysis. Common validated attacks still pending. |
| DistilBERT three-seed training (Experiment 40) | Completed | Seeds 42, 123, 2026; predominantly synthetic training/validation/testing. |
| DistilBERT cross-source robustness (Experiments 38, 41) | Completed **as exploratory pilots** | CEAS negative class and Nazario positive class are perfectly confounded with source. |
| URL-feature baseline on shared evaluation data | Pending | Separate from general presence/absence of a URL. |
| Generative LLM baseline | Assigned to teammates / pending integration | Must share evaluation cases, thresholds and attack rules. |
| Final human-validated benchmark | Pending | Required for primary conclusions. |

## Selected results — read the caveats

| Experiment / dataset | F1 | Recall | Legitimate false-positive rate |
|---|---:|---:|---:|
| TF-IDF + LR, CEAS random split (spam/ham, historical) | 0.9958 | 0.9945 | Not reported in this result file |
| TF-IDF + LR, CEAS group split (spam/ham, historical) | 0.9936 | 0.9908 | 0.0039 |
| TF-IDF + LR, synthetic test (n=316; threshold 0.44) | 0.8858 | 1.0000 | 0.2611 |
| MiniLM + LR, synthetic test (n=316) | 0.8858 | 1.0000 | 0.2611 |
| Frozen DistilBERT, SpaPhish (n=1395; exploratory) | 0.6223 | 0.7866 | 0.8163 |
| Exp. 40, synthetic test (mean of 3 seeds, n=316) | **1.0000** | 1.0000 | 0.0000 |
| Exp. 40, original exploratory set (mean, n=100) | 0.9778 | 0.9565 | 0.0000 |
| Exp. 41, cross-source **exploratory** clean pilot (mean of 3 seeds, n=160) | **0.6977** | 0.9833 | **0.8375** |

Historical evidence: CEAS-trained TF-IDF + LR recall on Nazario fell from 0.1113 to 0.0537 after a neutral-footer modification (90 newly missed positives). This is a completed sensitivity experiment, not a directly comparable final benchmark. See the tracked `external_robustness.csv` and `augmentation_defense.csv`.

For the cross-source pilot, 80 corpus-labeled positive emails come from **Nazario** and 80 corpus-labeled negatives from **CEAS-08**. Thus *source and class cannot be disentangled*. It would be misleading to present the 0.6977 F1 as a validated estimate of real-world phishing performance.

Three positive-only transformations were tested: `lexical_proxy` (42/80 positives actually changed), `neutral_hidden_html` (80/80), and `url_defang` (13/80). No detected-to-missed flips occurred in this pilot across the three seeds. These modifications **have not been independently certified to preserve attack intent**, and URL defanging deliberately changes clickability; this is **not** evidence that the detector resists real evasion attacks.

Public, non-identifying summaries: `results/metrics/exp38_pilot_summary_public.csv`, `exp40_three_seed_summary_public.csv` and `exp41_three_seed_summary_public.csv`.

## Repository map

```text
CYB5/
├── README.md
├── requirements.txt                 # classical analyses and shared tools
├── requirements-ml.txt              # optional PyTorch/Transformers environment
├── docs/
│   ├── EXPERIMENTAL_PROTOCOL.md
│   ├── CYB5_RESEARCH_ROADMAP.md       # corrected V2 roadmap
│   ├── TEAM_HANDOFF.md
│   └── PRIVACY_AND_LIMITATIONS.md
├── src/
│   ├── preprocessing/               # corpus audits and data preparation
│   ├── models/                      # earlier baseline training scripts
│   ├── attacks/                     # existing robustness scripts
│   └── evaluation/
│       ├── experiment38_distilbert_pilot.py
│       ├── check_ceas_raw_provenance.py
│       └── ...
├── notebooks/                       # DistilBERT, MiniLM and external evaluations
├── results/
│   ├── metrics/                     # *reviewed* public summaries only
│   └── figures/
├── scripts/                         # notebook-output sanitization
└── tests/                           # lightweight deterministic attack tests
```

## Setup (Windows CMD)

```bat
cd /d D:\CYB5
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
python -m unittest discover -s tests -v
```

**GPU experiments:** run `notebooks/CYB5_Experiment40_ONE_CLICK_FINAL.ipynb` and `notebooks/CYB5_Experiment41_3seeds_External_Robustness.ipynb` in **Google Colab with an appropriate GPU**. The notebooks handle Colab-specific installs, Google Drive mounting and backups; Python/CUDA dependency combinations in Colab can differ from Windows. In Colab, use the notebook installation cell instead of installing this repository's full Windows environment.

## Private data and model placement

Raw and processed email datasets, pilot manifests, original review notes and trained checkpoints are **not distributed through GitHub**. To reproduce them, use appropriately licensed/access-approved copies and the **original fixed splits**; don't invent substitute labels or regenerate an already-inspected test split while reporting comparable results.

- `data/raw/CEAS_08.csv` — original CEAS input (spam/ham semantics).
- `data/processed/ceas_clean.csv`, `nazario_external.csv` — locally derived inputs.
- `results/metrics/group_{train,validation,test}_ids.csv` — **private** split identifiers; required for the local Exp. 38 preparation and not committed.
- `data/audit/cyb5_experiment38_pilot/pilot_manifest.csv` — private 160-email pilot manifest.
- `models/distilbert_cyb5/` — local checkpoint for Exp. 38.
- Three Exp. 40 checkpoints — backed up in the author's private Google Drive folder; not in this repo.

**Do not open or click email links. Do not commit raw email content, local model weights, authentication tokens, sample IDs or private annotations.** The eight CEAS cases discussed in Experiment 37A remain unverified; don't publish AI annotations as human ground truth.

## Reproduce Experiment 38 locally (exploratory only)

After placing the approved private inputs in their documented local paths:

```bat
python src\evaluation\experiment38_distilbert_pilot.py --mode prepare
python src\evaluation\experiment38_distilbert_pilot.py --mode run --model-dirs models\distilbert_cyb5
```

Preparation is idempotent unless `--force` is explicitly passed. **Do not use `--force` to silently replace the original pilot** if comparing against previously reported metrics.

Experiment 40/41 notebooks reproduce the recorded **model training/evaluation pipeline** only if the fixed private datasets and matching saved checkpoints are provided. They must **not** be reinterpreted as a new unseen phishing benchmark. See [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md).

## Collaboration and deliverables

The team will compare **DistilBERT, TF-IDF + logistic regression, a URL-feature detector and a generative LLM** using the **same validated samples, the same transformations and consistent score definitions**. Read the corrected [research roadmap](docs/CYB5_RESEARCH_ROADMAP.md) and [team handoff](docs/TEAM_HANDOFF.md) before submitting additional model results. The handoff assigns the existing TF-IDF baseline for **reuse and common evaluation only**, not reimplementation. The reading list is a starting bibliography: the final 15–25 references must include a majority published within the last three years.

- **24 November 2026:** experimental milestone (comparisons, errors, ablations, documented limitations).
- **8 December 2026:** scientific article (8–12 pages, two columns), slide deck (12–15 slides), reproducible code and research disclosure.

## Data and reproducibility ethics

The previously inspected CEAS test set and SpaPhish evaluations must not be described as newly untouched. The 80-message CEAS label-audit included a human first pass and later AI assistance; no independent human–human kappa is yet supported. Model-generated annotations and raw email uploads to third-party services must be disclosed in research limitations.

**License:** no open-source license is assigned here; the project owner/team should agree on one before enabling redistribution beyond the repository's default copyright conditions. Underlying datasets and pretrained models have their own terms.
