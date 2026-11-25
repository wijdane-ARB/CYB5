# Résultats classiques — Exp42 (URL-only vs TF-IDF historique)

## Corpus et splits
- Pilote : `data/audit/cyb5_experiment38_pilot/pilot_manifest_ceas_only.csv`
  - 40 positifs CEAS_08 + 40 négatifs CEAS_08
  - Couverture URL : 70 % (pos) / 72 % (neg) → pas de confound
- Train URL models : `data/phishing_validation/processed/{train,val,test}.parquet`
- Fuite exacte avec pilote : 0 (preuve : `exp42_provenance.json`)
- Fuite proche : à vérifier — voir `near_duplicate_audit.csv`

## Modèles
| Modèle | Corpus | Seuil | Règle |
|---|---|---|---|
| URL_LR | phishing_validation | 0.50 | max_f1 sur validation |
| URL_RF (×3 seeds) | phishing_validation | 0.50 | max_f1 sur validation |
| TFIDF_HISTORIQUE | phishing_validation (templates) | **0.44** | historique (Wijdane) |

## Résultats clean (subset all)
| Modèle | F1 | Recall | FPR | Latence médiane (ms) |
|---|---|---|---|---|
| URL_LR | 0.553 | 0.525 | 0.375 | 0.55 |
| URL_RF | 0.497 ± 0.078 | 0.358 | 0.075 | 26.9 |
| TFIDF_HISTORIQUE | (à recalculer avec 0.44) | | | |

## Robustesse (ASR conditionnel)
| Condition | URL_LR ASR | URL_RF ASR | TFIDF ASR |
|---|---|---|---|
| lexical_proxy | 0.0 (n=3) | 0.0 (n=3) | à recalculer |
| neutral_hidden_html | 0.0 (n=21) | 0.0 (n=14) | à recalculer |
| url_defang | 0.0 (n=21) | 0.0 (n=14) | à recalculer |

## Traitement des messages sans URL
- `featurize` retourne un vecteur de zéros si aucune URL n'est extraite.
- Le modèle apprend ce pattern comme signal "pas d'URL" (corrélé à légitime dans `phishing_validation`).
- Dans le pilote CEAS-only, la couverture est équilibrée → ce pattern n'est pas un raccourci.
- `reverse_defang` est appliqué avant `extract_urls` → URLs defangées traitées comme canoniques.

## Preuves d'absence de fuite
- `exp42_provenance.json` → exact_overlap = {0, 0, 0}
- `results/metrics/near_duplicate_audit.csv` → à vérifier
- `url_only_training.json` → seuil choisi sur validation uniquement

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


### Correction 4 — Vérifier le recouvrement proche (10 min)

Le script `near_duplicate_audit.py` existe dans ton tree. Lance-le sur le pilote CEAS vs `phishing_validation` :

```powershell
python src\evaluation\near_duplicate_audit.py --help

## Canonicalisation des URLs (reverse_defang)
- `url_features.reverse_defang()` : normalise `hxxp(s)://` → `http(s)://`, `[.]` → `.`, `[:]` → `:`, `[@]` → `@`.
- Appliqué avant `extract_urls()` dans toutes les évaluations.
- **Impact** : ASR `url_defang` passe de 1.00 (sans canonicalisation) à 0.00 (avec).
- **Interprétation honnête** : la vulnérabilité initiale n'était pas dans le modèle URL mais dans l'extracteur. C'est un résultat de prétraitement, pas de robustesse du modèle.
- **Risque résiduel** : une obfuscation Unicode (homoglyphes, zero-width) contournerait `reverse_defang`. Non testé ici.