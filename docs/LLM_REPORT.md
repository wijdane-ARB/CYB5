# Détecteur LLM génératif local (zéro-shot) — rapport (modèle à compléter après l'exécution)

> **Statut : pilote exploratoire à étiquettes de corpus.** 80 positifs Nazario et 80 négatifs CEAS-08 : classe et source sont parfaitement confondues, les étiquettes ne sont pas vérifiées manuellement, et aucune transformation n'est certifiée préserver l'intention d'attaque. Ces chiffres ne constituent **pas** une évaluation du phishing réel.

## 1. Configuration figée

| Élément | Valeur |
|---|---|
| Modèle | `qwen2.5:7b-instruct` — version/digest : `<llm_run_record.json → model_digest>` |
| Famille / taille / quantification | `<model_details>` (Ollama) |
| Licence | Apache-2.0 selon la fiche du modèle — **à revérifier** |
| Exécution | Ollama `<ollama_version>`, loopback uniquement, requêtes séquentielles (batch = 1) |
| Matériel | `<hardware.gpu>` |
| Paramètres | température 0, seed 42 (sans effet à T = 0), `num_ctx` 8192, `num_predict` 32, pas de troncature, sortie JSON non contrainte |
| Prompt | `configs/llm_prompt_v1.txt`, SHA-256 `08954d3d…aca075` ; aucun exemple de test, aucun label |
| Règle d'erreur | toute sortie invalide / refus / erreur / timeout → **LEGITIMATE** (sensibilité à la règle inverse fournie dans le CSV) |
| Données | les 160 `sample_id` du pilote Exp. 38/41 ; transformations importées de `experiment38_distilbert_pilot.py` (empreinte du code dans le run record) |

Un échauffement du modèle précède l'évaluation ; le chargement est exclu des latences.

## 2. Résultats (`results/metrics/llm_aggregate.csv`)

*À coller depuis le CSV : par condition, TP/FP/TN/FN, précision, rappel, F1, FPR (IC de Wilson), ΔF1 vs propre, positifs éligibles/modifiés, ASR avec **dénominateur**, flips détecté→manqué (et manqué→détecté), taux de sortie invalide, latence moyenne/médiane/p95.*

| Condition | Éligibles | F1 | Rappel | FPR | ΔF1 | Flips détecté→manqué / dénominateur | Invalides | Latence moy. (s) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean | – | | | | – | – | | |
| lexical_proxy | 42 attendus | | | | | | | |
| neutral_hidden_html | 80 attendus | | | | | | | |
| url_defang | 13 attendus | | | | | | | |

Contrôles à reporter : `unchanged_text_pred_diffs` (messages **non modifiés** dont la prédiction diffère de la condition propre : mesure le bruit non déterministe du LLM), `context_overflow_n`, `lenient_parse_n`, `refusal_n`.

## 3. Validation et injection de prompt

Résumé de `llm_validation_report.json` (14 messages synthétiques inertes, disjoints du pilote) : taux d'invalides `<…>`, cas d'injection corrects `<x/2>`. Un message du pilote contenant un motif d'injection apparent doit être signalé ici par son **nombre**, sans citation.

## 4. Analyse d'erreurs (5 FP et 5 FN)

À rédiger **sans reproduire de texte** : lire les messages localement (accès privé, IDs dans `llm_predictions_PRIVATE.csv`) puis décrire la cause plausible par catégorie (ex. « notification légitime avec lien et ton urgent », « phishing sans lien, formulation neutre »), en distinguant sortie invalide / refus / erreur de classement.

## 5. Constat, limite, lecture prudente

- **Constat (1 phrase, chiffré)** : …
- **Limite principale** : pilote confondu par la source ; les écarts mesurés peuvent refléter des différences de corpus (en-têtes, formatage, époque) plutôt que du phishing. 160 messages répétés sur 4 conditions ne sont pas 640 observations indépendantes ; la baisse ou l'absence de baisse de F1 après une attaque de 13 ou 42 messages ne permet pas de conclure à une robustesse générale.
- **URL defanging** : change volontairement la cliquabilité ; c'est un test de représentation, pas une évasion exploitable.
- **Déterminisme** : T = 0 ne garantit pas une identité bit à bit sur GPU ; voir `unchanged_text_pred_diffs`.
- **Latence** : liée au matériel indiqué ; ne pas la comparer à des modèles mesurés sur un autre matériel sans le signaler.

## 6. Reproduction

Voir `docs/LLM_OLLAMA_SETUP.md`. Tests hors ligne : `python -m unittest discover -s tests -v`.
