# Résultats classiques — Exp42 (URL-only vs TF-IDF historique)

## Corpus et splits
- Pilote : `data/audit/cyb5_experiment38_pilot/pilot_manifest_ceas_only.csv`
  - 40 positifs CEAS_08 + 40 négatifs CEAS_08
  - Couverture URL : 72.5 % (neg) / 70 % (pos) → pas de confound
- Train URL models : `data/phishing_validation/processed/{train,val,test}.parquet`
- Fuite exacte avec pilote : 0 (preuve : `exp42_provenance.json`)
- Fuite proche : non évaluée (limitation documentée, cf. §Limitations)

## Modèles
| Modèle | Corpus | Seuil | Règle |
|---|---|---|---|
| URL_LR | phishing_validation | 0.50 | max_f1 sur validation |
| URL_RF (×3 seeds) | phishing_validation | 0.50 | max_f1 sur validation |
| TFIDF_HISTORIQUE | phishing_validation (templates) | **0.44** | historique (Wijdane) |

## Résultats clean (subset all, n=80)
| Modèle | F1 | Recall | FPR | Latence médiane (ms) |
|---|---|---|---|---|
| URL_LR | 0.553 | 0.525 | 0.375 | **0.693** |
| URL_RF | **0.497 ± 0.078** | 0.358 ± 0.072 | **0.075** | **28.21** |
| TFIDF_HISTORIQUE | **0.644** | **0.700** | **0.475** | **0.829** |

## Robustesse (ASR conditionnel, subset all)
| Condition | URL_LR ASR | URL_RF ASR | TFIDF ASR |
|---|---|---|---|
| lexical_proxy | N/A (n=0, couverture 7.5 %) | N/A (n=0, couverture 52.5 %) | **0.00 (n=2)** |
| neutral_hidden_html | **0.00 (n=21)** | **0.00 (n=14.3)** | **0.179 (n=28)** |
| url_defang | **0.00 (n=21)** | **0.00 (n=14.3)** | **0.700 (n=20)** |

## Traitement des messages sans URL
- `featurize` retourne un vecteur de zéros si aucune URL n'est extraite.
- Dans `phishing_validation`, ce pattern est corrélé à "légitime" (49 % des messages ont une URL).
- Dans le pilote CEAS-only, la couverture est équilibrée (70 % pos / 72.5 % neg) → ce pattern n'est pas un raccourci.
- Limitation : URL-only ne peut détecter aucun des 12 positifs sans URL → recall plafonné à 70 % maximum.

## Canonicalisation des URLs (reverse_defang)
- `url_features.reverse_defang()` : normalise `hxxp(s)://` → `http(s)://`, `[.]` → `.`, `[:]` → `:`, `[@]` → `@`.
- Appliqué avant `extract_urls()` dans toutes les évaluations.
- **Impact** : ASR `url_defang` passe de 1.00 (sans canonicalisation) à 0.00 (avec).
- **Interprétation honnête** : la vulnérabilité initiale n'était pas dans le modèle URL mais dans l'extracteur.
- **Risque résiduel** : une obfuscation Unicode (homoglyphes, zero-width) contournerait `reverse_defang`. Non testé ici.

## Preuves d'absence de fuite
- `exp42_provenance.json` → `exact_overlap = {train: 0, val: 0, test: 0}`
- `url_only_training.json` → seuil choisi sur validation uniquement
- `near_duplicate_audit.py` → non exécuté sur CEAS pilote (limitation)

## Limitations
- Pilot n=80 : intervalles de confiance larges (Wilson), non extrapolables.
- URL-only plafonné à recall=0.70 par construction (12 positifs sans URL).
- TF-IDF `phishing_template_lr` entraîné sur templates → non représentatif du réel.
- `lexical_proxy` n'affecte que 7.5 % des positifs → ASR sur cette condition non concluant.

## Reproduction
```powershell
# URL models
python src\models\train_url_features.py --train-source synthetic `
  --synthetic-dir data\phishing_validation\processed --out-dir models\url `
  --ablation --eval-test

# Évaluation
python src\evaluation\eval_classical_exp42.py `
  --manifest data\audit\cyb5_experiment38_pilot\pilot_manifest_ceas_only.csv `
  --url-models models\url\url_lr_seed42.joblib `
               models\url\url_rf_seed42.joblib `
               models\url\url_rf_seed123.joblib `
               models\url\url_rf_seed2026.joblib `
  --synthetic-dir data\phishing_validation\processed `
  --tfidf TFIDF_HISTORIQUE models\phishing_template_lr.joblib 0.44 synthetic


---

# ② Peut-on améliorer les résultats ? — OUI, mais avec des limites structurelles

## Comprendre **pourquoi** il y a encore beaucoup d'erreurs

Regarde les matrices de confusion (n=80) :

| Modèle | TN | FP | FN | TP | Erreurs |
|---|---|---|---|---|---|
| TFIDF_HISTORIQUE | 21 | 19 | 12 | 28 | **31/80** (39 %) |
| URL_LR | 25 | 15 | 19 | 21 | **34/80** (42.5 %) |
| URL_RF | 37 | 3 | 24 | 16 | **27/80** (34 %) |

### La limite structurelle (incontournable)

**URL_RF fait 24 FN sur 40 positifs.** Décomposons :
- 12 positifs **n'ont pas d'URL** → URL-only les prédit toujours 0 (faux négatifs **structurels**, aucune feature ne peut les sauver)
- 28 positifs **ont une URL** → URL_RF en détecte 16 (57 % de recall sur ce sous-ensemble)

Donc :
- **Recall max théorique = 28/40 = 70 %** pour URL-only (les 12 sans URL sont perdus d'avance)
- **Recall actuel = 16/40 = 40 %** → il y a **12 positifs avec URL manqués**

**C'est là que tu peux améliorer** : passer de 16/28 à ~22/28 sur les positifs avec URL (recall URL-subset 57 % → 79 %) ferait passer F1 de 0.497 à **~0.63**.

### La limite structurelle du corpus

Le modèle URL est entraîné sur **templates** (`phishing_validation`) où les hosts sont trivialement séparables (`secure-verify-team.site` vs `meet.company.com`). Sur CEAS (vrais emails), cette séparabilité n'existe plus → chute de F1 de 0.84 (test templates) à 0.497 (CEAS réel). **Aucune feature ne peut compenser un décalage de distribution de cette ampleur.**

---

## Améliorations par ordre d'impact

### 🟢 Amélioration 1 — Recalibrer le seuil (5 min, gain +0.05 à +0.10 F1)

Actuellement `thr=0.50 max_f1`. Le max_f1 optimise F1 sur validation templates, pas sur CEAS. Teste :

```powershell
# Test avec threshold-rule=max_recall_at_fpr max_fpr=0.10
python src\models\train_url_features.py `
  --train-source synthetic `
  --synthetic-dir data\phishing_validation\processed `
  --out-dir models\url_fpr10 `
  --threshold-rule max_recall_at_fpr --max-fpr 0.10 --ablation

