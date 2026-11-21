# Privacy, dataset provenance and research limitations

## Public repository safety

The project intentionally ignores `/data/`, `/models/`, ZIP archives and most `/results/`. Only aggregate, explicitly reviewed result summaries are allowlisted. Git ignores only **untracked** material: previously committed tables can remain publicly accessible. Prior to sharing, audit the repository history and existing tracked files, especially files containing `text_hash`, sample-level scores or linkable identifiers.

Do not publish:
- Full email messages, URL click targets, private pilot manifests, human reviewer sheets, raw provenance, personally identifying sender/recipient fields or secret keys.
- Google Drive authorization artifacts, tokens, local caches, credentials, `.env` files or model weight archives.
- Private prediction-level tables without an approved disclosure/consent policy.

Do not follow links or render active email HTML while reviewing potential phishing. Email text previously shared in ChatGPT/other AI environments **was transferred to those services**; don't claim an entirely local annotation workflow.

## Known evidence limitations

- CEAS-08 original positive labels are **spam**, not independently verified phishing.
- The CEAS 80-email audit had an initial human two-class annotation, an LLM-assisted reclassification suggestion, and unresolved cases; a second independent human four-way review and Cohen's kappa are **not yet available**.
- The Nazario-positive/CEAS-negative pilot completely confounds source and class. The external FPR/F1 values are diagnostic, not representative deployment estimates.
- SpaPhish is Spanish-language and was already inspected in earlier error analysis; the original CEAS test partition was also investigated earlier.
- Synthetic-v3 100% F1 on a template-separated set is reproducibility evidence, not proof of generalization.
- Exp. 41 attacks were not independently validated as meaning-preserving; URL defanging changes clickability. Positive-only attacks mechanically leave negative FPR unchanged.
- Final threshold selection must use a separate validation set. The threshold sweep on the pilot (Exp. 39) is post-hoc only.
- Experiment 40 last execution **reused** three previously saved checkpoints; original complete epoch-by-epoch logs are missing. We can report checkpoint provenance and per-seed predictions, but should not invent historical loss curves.

## Research disclosure text (adapt to faculty requirements)

> AI assistants were used for code drafting, debugging, documentation and preliminary email-label suggestions. The student's initial CEAS two-class review was human-conducted, but subsequent four-way proposals involved AI systems and were not treated as independent human ground truth. Selected email contents were shared with AI services during exploratory review. Reported cross-corpus and adversarial tests are exploratory, and verified labeling, attack-validity review and an unbiased held-out benchmark remain outstanding.
