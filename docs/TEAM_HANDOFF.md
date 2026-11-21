# CYB-5 — Fiche de répartition corrigée (V2)

**Question :** un détecteur d'hameçonnage fondé sur un modèle de langage résiste-t-il mieux à des messages réécrits pour l'évasion qu'un modèle classique sur variables extraites ?
**Jalon : 24/11/2026 ; final : 08/12/2026.**
**Roadmap détaillée :** `docs/CYB5_RESEARCH_ROADMAP.md`.

## Travail déjà réalisé par Wijdane (vérifié sur GitHub, commit b3f5978)

TF-IDF + LR sur CEAS, CEAS groupé, TREC et corpus synthétique ; ComplementNB ; analyses de fuite, seuils, ablations, attaques textuelles, augmentation ; MiniLM + LR ; DistilBERT ; SpaPhish ; audit préliminaire CEAS ; expériences 38–41. **Ne pas refaire ces travaux.** Le TF-IDF synthétique existant atteint F1 0,8858 (seuil 0,44), et constitue la baseline à réutiliser après contrôle de recouvrement. Les résultats historiques ne sont pas des scores comparables sur un benchmark final commun.

## À retenir avant de coder

- **Une comparaison = même ensemble de messages, mêmes attaques, mêmes métriques.** Le pilote actuel (80 Nazario positifs + 80 CEAS négatifs) est **exploratoire et confondu par la source** ; ne pas prétendre à un résultat phishing réel vérifié.
- CEAS est **spam/ham**, pas vérifié phishing. Ne pas utiliser le pilote comme calibration après examen ; ne pas entraîner TF-IDF sur des emails CEAS du pilote.
- Publier le code et **les agrégats anonymisés seulement** ; aucun corps d'email, vrai lien, manifest privé ou checkpoint sur GitHub public.

| Personne | À réaliser pour le 23/11 au soir | Livrer |
|---|---|---|
| **Wijdane — travaux réalisés et intégration** | Synthèse des baselines classiques, MiniLM, DistilBERT et Exp 37–41 ; expliquer 3 seeds, F1 synthétique=1,000, pilote F1 moyen=0,6977 et FPR=0,8375 ; vérifier cohérence des sorties de tous les modèles | Méthodes, résultats, limites, tableau consolidé |
| **Collègue 1 — LLM** | Choisir un LLM local, figer prompt JSON/paramètres, classer 160 messages × 4 conditions ; calculer F1/recall/FPR/ASR/latence et erreurs de parsing | Script, prompt, config, métriques agrégées, rapport 1 p. |
| **Collègue 2 — URL-only + intégration TF-IDF existant** | Construire seulement le nouveau détecteur URL ; réutiliser `train_phishing_templates.py` / checkpoint correspondant ; vérifier absence de fuite, seuils et mêmes IDs/attaques ; ablations URL, couverture et coût | Script URL-only, adaptateur d’évaluation TF-IDF, configs, métriques et ablation |
| **Collègue 3 — Benchmark/stats** | Définir labels, arbitrer 8 emails douteux, contrôler recouvrement, figer le manifeste privé, certifier l'intention des attaques, comparaisons appariées/IC | Fiche benchmark, protocole, validation, statistiques, limites |

## Format d'une remise acceptée

Chaque collègue partage : **(1) commit/PR, (2) commande de reproduction, (3) version/config, (4) CSV de métriques sans texte brut, (5) 1 page de résultats/erreurs/limites.** Les comparaisons doivent inclure F1 propre/attaqué, FPR, nombre d'emails réellement modifiés, ASR avec dénominateur, et temps/email.

## Séquence

- **21–22 novembre** : geler protocole, vérifier labels et commencer les scripts.
- **23 novembre** : terminer et confronter les quatre détecteurs ; ablations et analyse des erreurs.
- **24 novembre** : jalon expérimental, limites visibles.
- **25 novembre–7 décembre** : article 8–12 pages deux colonnes, 15–25 références justifiées, majorité de moins de trois ans, PPTX 12–15 slides, tests de reproduction.
- **8 décembre** : remise finale.

**Références prioritaires à lire en premier :** E-PhishGEN (AISec 2025), PhishBench 2.0 (CCS 2020), TextAttack (EMNLP 2020), TextFooler (AAAI 2020), DistilBERT (2019), Guo *On Calibration* (ICML 2017), WILDS (ICML 2021). La liste de départ annotée est dans la roadmap ; compléter avec des publications 2024–2026 vérifiées pour satisfaire la majorité récente exigée.

**Le pilote B est exploratoire.** Pour la comparaison finale, tester les quatre méthodes sur le nouveau benchmark C gelé et les variantes validées ; si C n'est pas prêt, déclarer le manque au J5. Ne pas considérer des répétitions d'inférence comme trois nouvelles graines d'entraînement.
