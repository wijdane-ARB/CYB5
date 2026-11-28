
# CYB-5 — Adversarial Robustness of Phishing Email Detection

**ENSET Mohammedia | Master's in Distributed Systems and Artificial Intelligence | Advanced AI Research Project**

## Research Question

**English:** Can a language-model-based phishing detector withstand evasion-oriented email rewrites compared with a classical feature-based detector?

**French:** Un détecteur d'hameçonnage fondé sur un modèle de langage résiste-t-il à des messages réécrits pour l'évasion, comparé à un modèle classique sur variables extraites ?

**Project status:** Research in progress. Historical classical and Transformer experiments are complete. The final common benchmark, URL-feature classifier, generative LLM comparison, and validated adversarial comparison remain incomplete.

This repository documents experimental research. It is **not a production-ready phishing detection system**.

---

## 1. Research Objectives

The CYB-5 project investigates whether language-model-based phishing detectors are more resistant to adversarial email transformations than classical feature-based classifiers.

The principal objectives are:

1. Measure classification performance on unmodified emails.
2. Compare classical machine learning and Transformer-based detection.
3. Evaluate classification changes following controlled email transformations.
4. Measure false-positive rates on legitimate emails.
5. Investigate generalization across datasets, sources, and languages.
6. Analyze model errors and possible dataset biases.
7. Report reproducibility, variability across independent training runs, and inference cost.

A successful experiment requires not only high classification performance but also reliable evaluation conditions and scientifically justified conclusions.

---

## 2. Models Implemented

| Model | Description | Status |
|---|---|---|
| TF-IDF + Logistic Regression | Classical lexical classifier | Completed |
| TF-IDF + Complement Naive Bayes | Classical probabilistic baseline | Completed |
| MiniLM + Logistic Regression | Frozen Transformer embeddings with a linear classifier | Completed |
| DistilBERT | Transformer fine-tuned for phishing classification | Completed |
| URL-feature classifier | Classifier using engineered URL characteristics | Pending collective integration |
| Generative LLM | Prompt-based phishing/legitimate classification | Pending collective integration |

### 2.1 TF-IDF + Logistic Regression

TF-IDF represents emails using weighted textual features.

Logistic regression classifies these representations using a learned decision boundary.

The implementation has already been evaluated on CEAS-08, grouped CEAS splits, TREC corpora, and the synthetic phishing benchmark.

Relevant scripts:

- `src/models/train_baseline.py`
- `src/models/train_group_baseline.py`
- `src/models/train_meajor_baseline.py`
- `src/models/train_phishing_templates.py`

The existing TF-IDF implementation should be reused for the common benchmark. It does not need to be implemented again.

The synthetic phishing classifier uses a historically validation-selected decision threshold of 0.44.

This threshold must not be optimized on an already-inspected evaluation set.

### 2.2 Complement Naive Bayes

Complement Naive Bayes provides a classical probabilistic baseline for text classification.

It was evaluated during the historical TREC experiments and compared against logistic regression.

Threshold selection and paired bootstrap comparisons were also investigated.

### 2.3 MiniLM + Logistic Regression

MiniLM is used as a frozen text encoder.

The resulting embeddings are supplied to a logistic regression classifier.

The Transformer encoder itself was not fine-tuned during this experiment.

MiniLM was evaluated on the synthetic phishing benchmark and on the exploratory original-email subset.

### 2.4 Fine-Tuned DistilBERT

DistilBERT was adapted to phishing classification using supervised fine-tuning on a predominantly synthetic English-language dataset.

Its pretrained language model is English-oriented.

The fine-tuned checkpoints were subsequently reused for external evaluations without further training.

**Important distinction:**

- Fine-tuning occurred during the training experiments.
- During SpaPhish inference, the previously fine-tuned checkpoint was loaded without retraining.
- The term "frozen during evaluation" does not mean the model had never been fine-tuned.

---

## 3. Datasets and Label Semantics

Different datasets were used for different scientific purposes. Their labels are not automatically interchangeable.

| Dataset | Size | Original labels | Purpose |
|---|---:|---|---|
| CEAS-08 | 39,100 cleaned emails | Ham / Spam | Classical baselines, leakage audits, sensitivity analyses |
| Nazario | 1,564 emails | Phishing-positive | External phishing recall and exploratory robustness |
| MeAJOR / TREC | Multiple corpora | Ham / Spam | Cross-corpus generalization and threshold analysis |
| Phishing Validation v3 | 2,000 emails | Phishing / Legitimate | Synthetic-heavy model training and testing |
| SpaPhish | 1,395 emails | Phishing / Legitimate | Exploratory external Spanish-language evaluation |

### 3.1 CEAS-08

The cleaned CEAS-08 dataset contains:

- 17,305 ham emails.
- 21,795 spam emails.
- 39,100 emails in total.

CEAS-08 identifies spam and legitimate email categories.

**Spam is not equivalent to independently verified phishing.**

Results on CEAS-08 must therefore be described as spam/ham classification unless additional independent phishing annotation has been performed.

### 3.2 Nazario

The Nazario corpus contains 1,564 phishing-labeled emails used in the project.

Because it contains no intrinsic legitimate comparison class, it cannot independently provide a false-positive rate or a complete binary-classification accuracy.

It is suitable for exploratory external phishing recall and positive-class sensitivity analyses.

### 3.3 MeAJOR / TREC

TREC-05, TREC-06, and TREC-07 were used to investigate cross-corpus classification and threshold calibration.

These corpora primarily concern spam/ham classification rather than independently verified phishing.

Their results should not be presented as equivalent to phishing/legitimate classification results.

### 3.4 Phishing Validation v3

The dataset contains 2,000 examples:

- 1,900 synthetic emails.
- 100 original emails retained separately.

The synthetic examples were divided by template families.

| Partition | Emails | Template families |
|---|---:|---:|
| Training | 1,307 | 36 |
| Validation | 277 | 8 |
| Test | 316 | 8 |
| Original exploratory | 100 | Separate original group |

The template-separated split limits direct reuse of the same template families between partitions.

However, this does not eliminate every form of shortcut learning, shared generation artifacts, or distribution bias.

The synthetic validation and test partitions have already been inspected.

They must not be described as newly untouched evaluation sets.

### 3.5 SpaPhish

SpaPhish contains 1,395 Spanish-language emails:

- 731 phishing.
- 664 legitimate.

It was used to investigate transfer from an English-trained phishing classifier to a different language and corpus source.

The SpaPhish evaluation has already been inspected and must be described as exploratory.

---

## 4. Experimental Results

All metrics below are historical recorded results.

They were obtained from different datasets and experimental protocols.

**Rows from different corpora must not be interpreted as one controlled model ranking.**

### 4.1 Selected Classification Results

| Experiment / Dataset | F1 | Recall | FPR |
|---|---:|---:|---:|
| TF-IDF + LR, CEAS random split | 0.9958 | 0.9945 | Not reported in this summary |
| TF-IDF + LR, CEAS grouped split | 0.9936 | 0.9908 | 0.0039 |
| TF-IDF + LR, synthetic test | 0.8858 | 1.0000 | 0.2611 |
| MiniLM + LR, synthetic test | 0.8858 | 1.0000 | 0.2611 |
| DistilBERT, SpaPhish external evaluation | 0.6223 | 0.7866 | 0.8163 |
| Experiment 40, DistilBERT synthetic test, three-seed mean | 1.0000 | 1.0000 | 0.0000 |
| Experiment 40, original exploratory set, three-seed mean | 0.9778 | 0.9565 | 0.0000 |
| Experiment 41, external exploratory pilot, three-seed mean | 0.6977 | 0.9833 | 0.8375 |

The near-identical TF-IDF and MiniLM results on the synthetic test reflect the recorded experimental outputs.

Both models produced the same aggregate confusion-matrix counts on this benchmark.

This numerical agreement must not be interpreted as proof that the architectures are equivalent or that their predictions necessarily agree on every individual example.

---

## 5. Experiment 31 — SpaPhish External Evaluation

### 5.1 Objective

Evaluate whether a previously fine-tuned English-language DistilBERT phishing classifier generalizes to Spanish-language emails without additional training.

### 5.2 Experimental Configuration

| Parameter | Value |
|---|---|
| Model | Previously fine-tuned DistilBERT |
| Evaluation dataset | SpaPhish |
| Evaluation language | Spanish |
| Samples | 1,395 |
| Decision threshold | 0.50 |
| Maximum sequence length | 256 tokens |
| Training during evaluation | None |
| Evaluation type | Cross-language and cross-source transfer |

The saved checkpoint was loaded and placed in evaluation mode.

No SpaPhish examples were used for additional model fine-tuning during Experiment 31.

Script:

`src/evaluation/evaluate_spaphish_distilbert.py`

### 5.3 Recorded Results

| Metric | Value |
|---|---:|
| Accuracy | 0.4996 |
| Precision | 0.5148 |
| Recall | 0.7866 |
| F1-score | 0.6223 |
| False-positive rate | 0.8163 |
| True negatives | 122 |
| False positives | 542 |
| False negatives | 156 |
| True positives | 575 |

Confusion matrix:

```text
                     Predicted
                   Safe   Phishing
Actual Safe         122       542
Actual Phishing     156       575
```

These values are consistent with the committed file:

`results/metrics/spaphish_distilbert_external.json`

### 5.4 Input Truncation

The evaluation used a maximum input length of 256 tokens.

Of the 1,395 SpaPhish emails, **758 exceeded this limit** and were truncated during inference.

This means potentially relevant content could have been excluded from the classifier input.

The historical training configuration also used a 256-token input limit.

Truncation is a possible explanatory factor, but its causal contribution has not been established.

### 5.5 Scientific Interpretation

SpaPhish represents a simultaneous change in:

- Language.
- Dataset provenance.
- Email style and content distribution.
- Potential phishing and legitimate-message characteristics.

Therefore, the performance degradation cannot be attributed to the Spanish language alone.

The experiment demonstrates limited performance under this particular external transfer condition.

It does not independently isolate language mismatch, domain shift, or truncation.

Additional controlled experiments would be required to distinguish these explanations.

---

## 6. Experiment 40 — Three-Seed DistilBERT Training

### 6.1 Objective

Evaluate whether the recorded DistilBERT performance is consistent across independent training seeds.

### 6.2 Training Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | `distilbert/distilbert-base-uncased` |
| Random seeds | 42, 123, 2026 |
| Training emails | 1,307 |
| Validation emails | 277 |
| Synthetic test emails | 316 |
| Maximum sequence length | 256 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Maximum training epochs | 3 |
| Weight decay | 0.01 |
| Selected epoch | 1 for each seed |
| Selected threshold | 0.50 for each seed |

The checkpoint-selection procedure used validation performance.

The validation threshold-selection procedure targeted a false-positive rate no greater than 0.05.

### 6.3 Recorded Results

| Partition | F1, seed 42 | F1, seed 123 | F1, seed 2026 |
|---|---:|---:|---:|
| Synthetic validation | 1.0000 | 1.0000 | 1.0000 |
| Synthetic test | 1.0000 | 1.0000 | 1.0000 |
| Original exploratory | 0.9778 | 0.9778 | 0.9778 |

All three checkpoints achieved perfect F1 on the synthetic validation and previously inspected synthetic test sets.

### 6.4 Interpretation and Limitations

Perfect results on both validation and test indicate that the synthetic classification task may be unusually easy or highly separable.

Possible explanations include:

- Repetitive generation patterns.
- Strong lexical differences between generated classes.
- Template-related shortcuts.
- Limited diversity in legitimate and phishing examples.

These are hypotheses, not independently demonstrated causes.

The perfect result does not establish general phishing-detection performance or adversarial robustness.

The original validation and synthetic test partitions were previously examined during development.

They should be treated as historical exploratory evaluations.

### 6.5 Training History and Reproducibility

Three separately trained checkpoints were archived and subsequently recovered for repeated evaluation.

The final Experiment 40 notebook execution reused those saved checkpoints rather than training all three models from scratch again.

The public exports identify epoch 1 as the selected checkpoint for each seed.

**The complete original per-epoch training-loss histories are not archived.**

Consequently, complete three-seed training-loss curves must not be presented as experimentally verified.

The existing checkpoints and historical metric exports should be preserved.

---

## 7. Experiments 38 and 41 — External Exploratory Pilot

### 7.1 Dataset Composition

The fixed pilot contains 160 emails:

- 80 phishing-labeled Nazario emails.
- 80 CEAS ham emails.

All positive examples come from Nazario.

All negative examples come from CEAS.

This creates **perfect confounding between the source dataset and the class label**.

Consequently, the pilot is not a representative independent real-world phishing benchmark.

### 7.2 Experiment 38

Experiment 38 used a previously fine-tuned DistilBERT checkpoint to evaluate classification behavior on the fixed pilot.

The evaluation threshold was 0.50.

The pilot was created for preliminary cross-source diagnostics and transformation testing.

### 7.3 Experiment 39 — Threshold Analysis

Experiment 39 examined alternative classification thresholds on the already-inspected pilot.

A threshold of 0.98 produced a higher apparent F1 for the tested configuration.

However, this was post-hoc analysis on an already-examined evaluation set.

Therefore, the threshold must not be presented as independently calibrated or selected on untouched validation data.

### 7.4 Experiment 41 — Three-Seed Evaluation

Experiment 41 reused three previously trained DistilBERT checkpoints.

Seeds:

- 42.
- 123.
- 2026.

Threshold:

- 0.50 for all three seeds.

Recorded clean results:

| Seed | F1 | Recall | FPR |
|---|---:|---:|---:|
| 42 | 0.6870 | 0.9875 | 0.8875 |
| 123 | 0.6840 | 0.9875 | 0.9000 |
| 2026 | 0.7222 | 0.9750 | 0.7250 |
| Mean | 0.6977 | 0.9833 | 0.8375 |

The sample standard deviation across seeds was approximately:

- F1: 0.0213.
- Recall: 0.0072.
- FPR: 0.0976.

The very high false-positive rate is consistent with generalization problems, but the pilot cannot identify their cause because source and class are perfectly confounded.

---

## 8. Adversarial Transformations and Robustness

### 8.1 Transformation Types

Three exploratory transformations were used.

| Transformation | Description | Positive emails modified |
|---|---|---:|
| `lexical_proxy` | Conservative lexical substitutions | 42/80 |
| `neutral_hidden_html` | Addition of neutral hidden HTML text | 80/80 |
| `url_defang` | Changes to URL representation | 13/80 |

These transformations were applied to the positive examples in the exploratory pilot.

The negative examples were left unchanged.

Therefore, the false-positive rate was unchanged across attack conditions by experimental design.

### 8.2 Recorded Observations

Experiment 41 did not observe any transitions from correctly detected phishing to missed phishing under the tested transformations.

However, the baseline phishing recall was already between 0.9750 and 0.9875.

Only one or two of the 80 positive examples were missed before transformation, depending on the seed.

This means the classifier already predicted the phishing class for nearly every positive example.

The observed absence of additional misses is therefore weak evidence of robustness.

### 8.3 Attack Validity Limitations

The current transformations are better described as **exploratory sensitivity probes** than fully validated adversarial attacks.

In particular:

- Lexical substitutions were not independently certified as semantic paraphrases.
- Neutral hidden HTML tests sensitivity to added content.
- URL defanging can remove clickability and may alter malicious functionality.

A valid evasion attack must preserve the relevant malicious objective while changing the representation observed by the detector.

A final adversarial claim requires verification of transformation eligibility and preservation of attack intent.

### 8.4 Historical Classical Robustness Tests

Additional classical-model experiments investigated:

- Changes in capitalization.
- Changes in punctuation.
- Whitespace modifications.
- Neutral footer insertion.
- URL and lexical masking.
- Data augmentation.
- Cross-corpus changes.

For a CEAS-trained TF-IDF classifier evaluated on Nazario, adding a neutral footer decreased recorded phishing recall from approximately 0.1113 to 0.0537.

This demonstrates sensitivity under the tested conditions.

It is not a final controlled comparison with DistilBERT or a generative LLM.

---

## 9. Data Quality and Annotation

### 9.1 Experiment 37A — CEAS Label Audit

An exploratory review was performed on 80 CEAS emails sampled from the grouped training and validation partitions.

The initial human review distinguished ham and spam.

Subsequent AI-assisted assessments proposed more specific categories, including possible phishing.

Those AI-generated suggestions are not equivalent to independent human ground truth.

The proposed phishing cases remain unverified because sender authenticity and malicious destination information were not established.

No independent human-human inter-rater agreement statistic is currently supported.

### 9.2 Required Label Definitions

For the final binary benchmark:

- `1` = phishing, with documented supporting evidence.
- `0` = legitimate, with documented supporting evidence.

Ordinary spam and uncertain cases must not automatically be assigned phishing labels.

Each record should retain its original label provenance.

Independent review and adjudication are necessary where original labels are insufficient.

---

## 10. Final Common Benchmark — Work Remaining

The historical experiments do not yet provide a fully controlled comparison answering the central research question.

The final study requires a common benchmark for:

1. TF-IDF + Logistic Regression.
2. Fine-tuned DistilBERT.
3. A dedicated URL-feature classifier.
4. A generative LLM classifier.

All participating models must be evaluated on the same approved emails and transformations.

### 10.1 Benchmark Requirements

The shared evaluation must use:

- Fixed sample identifiers.
- Documented labels and label provenance.
- Identical original email content.
- Identical precomputed transformation variants.
- Fixed train, validation, and test partitions.
- Exact and near-duplicate overlap checks.
- Thresholds selected without using final evaluation labels.
- Frozen model checkpoints during the final evaluation.
- A documented threat model.
- Independent validation of adversarial transformation validity.

The historical 160-email Nazario/CEAS pilot must not be presented as the independently validated final benchmark.

### 10.2 Recommended Evaluation Metrics

For each model and experimental condition, report:

- Precision.
- Recall.
- F1-score.
- False-positive rate.
- Confusion matrix.
- Mean and standard deviation over independent seeds where applicable.
- Attack eligibility and transformation coverage.
- Attack success rate among initially detected, validly attacked phishing emails.
- Mean inference latency.
- Hardware and batch-size conditions.
- Paired uncertainty estimates where appropriate.

The final comparison must distinguish clean performance from degradation caused by valid adversarial modifications.

---

## 11. Repository Structure

```text
CYB5/
├── README.md
├── requirements.txt
├── requirements-ml.txt
├── .gitignore
├── data/                         # Private, ignored by Git
├── models/                       # Private, ignored by Git
├── docs/
│   ├── EXPERIMENTAL_PROTOCOL.md
│   ├── CYB5_RESEARCH_ROADMAP.md
│   ├── TEAM_HANDOFF.md
│   └── PRIVACY_AND_LIMITATIONS.md
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── attacks/
│   └── evaluation/
│       ├── evaluate_spaphish_distilbert.py
│       ├── experiment38_distilbert_pilot.py
│       └── check_ceas_raw_provenance.py
├── notebooks/
├── results/
│   ├── metrics/
│   └── figures/
├── scripts/
└── tests/
```

Historical files may still be tracked outside the current explicitly sanitized result summaries. See the privacy section before publishing additional results.

---

## 12. Installation and Environment

### 12.1 Windows CMD

From the project directory:

```bat
cd /d D:\CYB5
```

Create a virtual environment if one does not already exist:

```bat
python -m venv .venv
```

Activate the environment:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
```

Run tests:

```bat
python -m unittest discover -s tests -v
```

### 12.2 Google Colab

GPU experiments were conducted using Google Colab.

Relevant notebooks include:

- `notebooks/CYB5_Experiment40_ONE_CLICK_FINAL.ipynb`
- `notebooks/CYB5_Experiment41_3seeds_External_Robustness.ipynb`

Google Colab dependencies and hardware may differ from the local Windows configuration.

Use the notebook-specific installation instructions.

Saved checkpoints and fixed private input data are required to reproduce historical results faithfully.

The notebooks must not be presented as producing a new untouched evaluation dataset merely because they can be rerun.

---

## 13. Experiment Reproduction

### 13.1 Experiment 31 — SpaPhish

Script:

```bat
python src\evaluation\evaluate_spaphish_distilbert.py
```

Required local resources include:

- The approved SpaPhish CSV in `data/external/spaphish/`.
- The saved DistilBERT checkpoint in `models/distilbert_cyb5/`.
- The historical metrics archive `results/CYB5_DistilBERT_Results.zip`.

Running the script performs inference and rewrites the local metrics JSON.

Do not rerun solely to modify metadata when the historical results have already been archived.

Retain the original result JSON separately before conducting additional evaluations.

### 13.2 Experiment 38

Prepare the fixed pilot using the documented private inputs:

```bat
python src\evaluation\experiment38_distilbert_pilot.py --mode prepare
```

Evaluate the saved checkpoint:

```bat
python src\evaluation\experiment38_distilbert_pilot.py --mode run --model-dirs models\distilbert_cyb5
```

Do not use `--force` to silently replace the original pilot.

### 13.3 Experiments 40 and 41

Use the original notebooks with the corresponding saved checkpoints and fixed datasets.

The three Experiment 40 checkpoints were saved separately.

Historical training and evaluation results must be distinguished from new experiment runs.

---

## 14. Privacy, Data Handling, and Git History

Raw email content, private annotations, original pilot manifests, and trained model checkpoints must not be committed to the public repository.

Relevant private paths include:

- `data/raw/`
- `data/processed/`
- `data/external/`
- `data/audit/cyb5_experiment38_pilot/pilot_manifest.csv`
- `models/`
- Private experimental prediction files and split identifiers.

### 14.1 Historical Hash-Based Audit Files

The following historical files contained email-derived SHA-256 identifiers:

- `results/metrics/near_duplicate_audit.csv`
- `results/metrics/group_near_duplicate_audit.csv`

These files are being removed from Git tracking in the current correction branch while private local backups are retained.

Their historical Git revisions may remain accessible.

Removing a tracked file in a later commit does not erase it from Git history.

Any history cleanup must be planned separately, especially because other team members are working on shared branches.

### 14.2 Sanitized Public Results

The following explicitly reviewed summary files contain aggregate metrics rather than individual email records:

- `results/metrics/exp38_pilot_summary_public.csv`
- `results/metrics/exp40_three_seed_summary_public.csv`
- `results/metrics/exp41_three_seed_summary_public.csv`

The claim that these three summary files contain no email-derived hashes applies specifically to these files.

It does not establish that every historical tracked file or commit is free from potentially identifying information.

### 14.3 Research Data Precautions

- Do not publish raw private email content.
- Do not expose functional phishing URLs.
- Do not open suspicious email links.
- Do not publish access tokens or credentials.
- Do not treat automated annotations as verified human labels.
- Preserve private original experiments and checkpoints for reproducibility.
- Review notebooks for embedded outputs before publication.
- Audit tracked files and Git history before claiming full data sanitization.

Historical use of AI-assisted annotation or external tools must be accurately disclosed.

---

## 15. Scientific Limitations

The main limitations of the current experimental record are:

**Synthetic dataset simplicity:** Perfect validation and test results may reflect generation patterns or unusually separable classes.

**Label differences:** CEAS and TREC primarily represent spam/ham classification, while Nazario and SpaPhish provide different phishing-related label semantics.

**Source-class confounding:** In the 160-email pilot, Nazario supplies all positive examples and CEAS supplies all negative examples.

**Cross-language uncertainty:** The SpaPhish experiment simultaneously changes language and corpus source.

**Truncation:** Many SpaPhish messages exceed the 256-token input limit.

**Exploratory attacks:** Existing lexical, HTML, and URL transformations have not been independently certified as valid intent-preserving attacks.

**Training provenance:** Complete original per-epoch training-loss histories for all three Experiment 40 runs are unavailable.

**Inspected evaluation sets:** Several historical test partitions were previously examined during model development.

**Incomplete collective evaluation:** A leakage-controlled shared benchmark and final paired comparison are still required.

These limitations must be presented alongside the results rather than hidden behind aggregate F1-scores.

---

## 16. Collaboration and Responsibilities

The team is working toward one shared experimental comparison.

### Existing Work — Wijdane

Historical completed research includes:

- Corpus preprocessing and quality audits.
- CEAS and TREC baselines.
- TF-IDF and ComplementNB.
- MiniLM + Logistic Regression.
- DistilBERT fine-tuning.
- SpaPhish external evaluation.
- Experiments 37A–41.
- Classical robustness and ablation analyses.
- Research documentation and integration.

### Generative LLM Work

The generative LLM contribution should use a fixed prompt, fixed inference configuration, consistent labels, and the common benchmark.

Its outputs should be evaluated using the same clean and transformed messages as the other models.

### URL-Feature Classifier

The URL classifier is a separate classical feature-based model.

Previous URL masking experiments do not replace this implementation.

Its evaluation must use the common benchmark and account for messages without eligible URL features.

### Benchmark Validation and Statistics

The final benchmark requires source-aware sampling, label verification, leakage checks, transformation validation, and paired statistical analysis.

All contributors must share compatible output formats and evaluation definitions.

---

## 17. Research Deliverables

The project roadmap records the following milestones:

- Experimental milestone: 24 November 2026.
- Final project milestone: 8 December 2026.

Planned deliverables include:

- Scientific article.
- Presentation slides.
- Reproducible source code.
- Experimental metric summaries.
- Limitations and error analysis.
- Scientific references.
- Accurate AI-assistance disclosure.

**Document-length clarification:** The project roadmap previously specifies an 8–12-page article, while a separate methodology-of-research guidance document specifies 6–8 pages in IEEE format. The applicable requirement must be confirmed with the instructor before final submission.

---

## 18. Research Integrity

All reported results must be traceable to the relevant experiment, dataset, checkpoint, configuration, and evaluation script.

The project must not claim:

- CEAS spam labels are independently verified phishing labels.
- A perfect synthetic F1-score demonstrates real-world phishing security.
- The 160-email source-confounded pilot is a representative independent test.
- SpaPhish performance degradation was caused solely by Spanish-language mismatch.
- The absence of detected-to-missed flips proves adversarial robustness.
- Thresholds optimized on previously inspected tests were independently calibrated.
- Complete training histories exist when they are not archived.
- AI-assisted labels are equivalent to independent human ground truth.

All experimental limitations and the actual use of AI tools must be disclosed accurately.

---

## 19. Project Status and Next Steps

### Completed

- Classical TF-IDF and ComplementNB experiments.
- MiniLM synthetic benchmark evaluation.
- Fine-tuned DistilBERT experiments.
- Three independent DistilBERT checkpoints.
- Historical external evaluation on SpaPhish.
- Exploratory adversarial sensitivity tests.
- Historical error, leakage, and ablation analyses.

### Remaining

- Validate a stronger common phishing/legitimate benchmark.
- Confirm label provenance and dataset overlap.
- Integrate the URL-feature classifier.
- Integrate the generative LLM comparison.
- Reuse existing TF-IDF and DistilBERT models on the same benchmark.
- Evaluate identical verified adversarial transformations.
- Report per-model clean and attacked metrics.
- Add statistical uncertainty and inference-cost comparisons.
- Consolidate the scientific manuscript and presentation.

The existing experiments provide valuable evidence about classification behavior, synthetic-data limitations, and external generalization.

The final scientific conclusion must be based on the shared, independently validated comparison.

---

## 20. Repository and References

Project repository:

https://github.com/wijdane-ARB/CYB5

Research roadmap:

`docs/CYB5_RESEARCH_ROADMAP.md`

Experimental protocol:

`docs/EXPERIMENTAL_PROTOCOL.md`

Team handoff:

`docs/TEAM_HANDOFF.md`

Privacy and limitations:

`docs/PRIVACY_AND_LIMITATIONS.md`

The final article bibliography will contain the verified scientific references relevant to phishing detection, classical classifiers, language models, adversarial robustness, and cross-domain evaluation.

The research team is responsible for verifying reference accuracy, experimental reproducibility, and consistency between the manuscript and repository.

**License:** No open-source license has been assigned to this repository. Third-party datasets and pretrained checkpoints remain subject to their respective licenses and usage conditions.
