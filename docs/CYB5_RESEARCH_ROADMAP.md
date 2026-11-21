---
title: "CYB-5 — Feuille de route scientifique et opérationnelle"
subtitle: "Robustesse adversariale de la détection d’hameçonnage par modèles de langage et variables extraites"
date: "21 novembre 2026"
lang: fr
geometry: margin=0.8in
fontsize: 10pt
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# CYB-5 — Feuille de route scientifique et opérationnelle

**ENSET Mohammedia · Master Systèmes Distribués et IA · Version corrigée et vérifiée : 21 novembre 2026 (commit b3f5978)**
**Jalon expérimental : 24 novembre 2026 · Livrables finaux : 8 décembre 2026**
**Dépôt de code :** [github.com/wijdane-ARB/CYB5](https://github.com/wijdane-ARB/CYB5)
**Statut :** TF-IDF + LR, ComplementNB, MiniLM + LR et DistilBERT déjà implémentés et évalués ; comparaison commune finale, URL-only et LLM génératif restant à compléter. Cette version remplace la répartition précédente ; ne pas repartir de zéro.

> **Question de recherche donnée par l'enseignant :** « Un détecteur d'hameçonnage fondé sur un modèle de langage résiste-t-il à des messages réécrits pour l'évasion, comparé à un modèle classique sur variables extraites ? »

## 1. Ce que nous devons vraiment démontrer

Ce projet n'est **ni une application web ni une compétition pour obtenir l'accuracy la plus élevée**. C'est une **comparaison expérimentale contrôlée** : mêmes messages, mêmes attaques, mêmes métriques, séparation stricte entre apprentissage et évaluation. Nous voulons mesurer si les méthodes à base de représentations de langage perdent moins de performance que les méthodes classiques sous modification malveillante **tout en conservant une faible proportion de faux positifs sur les emails légitimes**.

### Questions opérationnelles (RQ)

| ID | Question mesurable | Mesure principale | Pourquoi elle répond à la question centrale |
|---|---|---|---|
| **RQ1** | Quelle performance obtient chaque détecteur **sur les emails propres** ? | F1 phishing, rappel, précision, FPR légitime | Une résistance apparente aux attaques n'a aucun sens si le modèle classe tout comme phishing. |
| **RQ2** | Comment la performance évolue-t-elle après une **réécriture d'évasion valide** ? | ΔF1, Δrappel ; *Attack Success Rate* (ASR) conditionnel | C'est le cœur de « résiste-t-il à des messages réécrits ? ». |
| **RQ3** | Quelle est l'influence du **type d'attaque** et de l'information exploitée ? | Comparaisons paraphrase / HTML / URL ; ablation | Distingue robustesse lexicale, structurelle et liée aux URL. |
| **RQ4** | La comparaison tient-elle sous **changement de domaine/source/langue** et à travers les graines ? | Résultats par source ; moyenne ± écart-type, IC | Empêche de confondre mémorisation de gabarits avec détection réelle. |
| **RQ5** | Quel est le **coût opérationnel** d'une défense ? | ms/email, matériel, débit, jetons, coût monétaire si API | Un détecteur efficace mais trop coûteux peut ne pas être déployable. |

### Hypothèses : à tester et non à affirmer

- **H1** : une représentation contextuelle (DistilBERT ou LLM) pourrait être moins sensible aux substitutions lexicales qu'un TF-IDF, **à FPR comparable**.
- **H2** : un modèle URL-only risque d'être très sensible aux obfuscations **sur les emails contenant effectivement une URL**, mais n'a aucun signal URL sur les messages sans lien.
- **H3** : l'excellente performance sur les gabarits synthétiques ne se transfère pas nécessairement aux courriels d'autres sources.
- **H4** : le choix du seuil et la provenance des données influencent davantage la conclusion que de faibles écarts de F1.

**Critère d'une bonne conclusion :** nous présentons les résultats, les incertitudes, l'éligibilité des attaques et les conditions dans lesquelles une méthode est supérieure ou inférieure. Si les données ne permettent pas une conclusion causale, nous le disons.

## 2. Justification précise de chaque choix de modèle

| Modèle | Rôle dans l'expérience | Pourquoi ce choix, au lieu d'une alternative ? | Ce que l'on doit contrôler | Responsable |
|---|---|---|---|---|
| **TF-IDF + régression logistique** | Baseline textuelle classique | Simple, rapide, interprétable ; mesure la dépendance aux mots et n-grammes. L'évaluation commune montre si un encodeur apporte un gain réel. | Même train/validation, seuil choisi **uniquement** sur validation, déduplication ; vérifier que le test ne chevauche pas le train CEAS. | Wijdane (déjà réalisé) ; collègue 2 (réutilisation pour la comparaison commune) |
| **Classificateur sur variables URL** | Baseline *sur variables extraites*, explicitement demandée | Isole les signaux structuraux (longueur URL, nombre de chiffres, hôte, chemins, caractères spéciaux) et teste la vulnérabilité aux obfuscations. | Ne jamais consulter l'URL en ligne ; prendre uniquement les caractéristiques disponibles dans le contenu ; compter les emails avec et sans URL ; ablation text-only/URL-only si possible. | Collègue 2 |
| **DistilBERT fine-tuné** | Détecteur fondé sur un modèle de langage, côté Wijdane | Encodeur contextuel relativement léger et déjà entraîné dans CYB-5 ; **trois seeds** disponibles. Permet de distinguer généralisation, stabilité et robustesse sans changement d'architecture. | Checkpoints gelés, max 256 tokens, seuils de validation, corpus synthétique à gabarits, langue et troncature. | Wijdane |
| **LLM génératif local (p. ex. Ollama)** | Comparaison « compréhension/prompt » sans fine-tuning | Peut exploiter des consignes générales et répondre à des variations textuelles ; fournit un point de comparaison avec le classificateur fine-tuné. | Prompt identique et figé, température 0 si disponible, sortie JSON stricte, exemples de prompts préalablement figés, taux de sortie invalide, version exacte et matériel. | Collègue 1 |

**Pas de course au meilleur score :** choisir l'unique modèle ou seuil après avoir vu le test invaliderait la comparaison. Si le temps manque, **quatre modèles simples et comparables** valent mieux qu'une pile de modèles complexes évalués sur des ensembles différents.

## 3. État vérifié du projet — expériences déjà effectuées

| Expérience | Résultat disponible | Interprétation honnête | Ce qui reste |
|---|---|---|---|
| **TF-IDF + LR — CEAS, split aléatoire et groupé** | F1 test 0,9958 et 0,9936 ; rappel Nazario respectivement 0,1113 et 0,1304 | Baselines déjà entraînées ; CEAS positif = spam/ham, pas phishing vérifié. Les évaluations sont déjà examinées. | Réutiliser le code et les checkpoints privés si disponibles ; contrôle de fuite pour le nouveau benchmark. |
| **TF-IDF + LR et ComplementNB — TREC/MeAJOR** | Modèles entraînés sur TREC-05, seuils sélectionnés sur TREC-06 : 0,69 (LR), 0,64 (NB) ; comparaison appariée 1 000 bootstraps sur TREC-07 | Résultats spam/ham exploratoires, pas preuve de phishing. | Aucune répétition des expériences historiques ; intégrer comme résultats secondaires. |
| **Attaques classiques et défense** | Casse, espacement, ponctuation, pied de page neutre ; rappel Nazario 0,1113 → 0,0537, 90 nouveaux faux négatifs ; augmentation testée | Sensibilité textuelle démontrée pour la baseline historique, pas comparaison harmonisée finale. | Réutiliser scripts et métriques ; tester les variantes communes contrôlées. |
| **Ablations et audit des fuites** | Masquage URL/email/nombres conjoint, longueur, proximité MinHash et analyses de seuil | Le masquage conjoint ne remplace pas un classificateur URL-only. | Réutiliser les analyses ; documenter séparément toute nouvelle ablation. |
| **TF-IDF + LR / MiniLM + LR — synthétique** | F1 0,8858, accuracy 0,8703 et FPR 0,2611 sur 316 messages pour chacun | Comparaison déjà effectuée ; 41 faux positifs ; domaine principalement synthétique. | Réutiliser le TF-IDF synthétique existant (`train_phishing_templates.py`) sur les variantes communes après vérification de recouvrement. MiniLM facultatif. |
| **DistilBERT et SpaPhish** | Fine-tuning initial et évaluation externe SpaPhish : F1 0,6223, FPR 0,8163 sur 1 395 emails ; fenêtres d'entrée et IC Wilson étudiés | Changement de langue/source ; SpaPhish déjà consulté. | Documenter sans réentraîner ni prétendre à un test indépendant. |
| **37A — Audit CEAS** | 80 emails échantillonnés depuis train/validation (40 de chaque étiquette originale) ; 48 `ham`, 32 `spam` au premier passage ; un autre LLM a suggéré 8 `PHISHING`, 24 `SPAM` | CEAS est initialement un corpus spam/ham : l'étiquette positive n'est **pas** une preuve de phishing ; les huit cas restent contestés, et les annotations IA ne sont pas des vérités terrain indépendantes. | Arbitrage humain des cas incertains et second annotateur indépendant si des affirmations sur la qualité des labels sont publiées. |
| **38 — Pilote adversarial DistilBERT** | 160 messages : 80 Nazario étiquetés phishing + 80 CEAS étiquetés ham ; F1 = **0,6870**, rappel = **0,9875**, FPR = **0,8875** ; trois attaques limitées | Diagnostic seulement : le label est confondu avec la source ; une FPR énorme empêche de qualifier le modèle de robuste. | Conserver comme expérimentation exploratoire et l'expliquer. |
| **39 — Sensibilité au seuil** | Seuil 0,50 : F1 0,687 ; 0,98 : F1 0,812 ; 0,99 : F1 0,792 sur le pilote inspecté | Illustration de la sensibilité au seuil, **pas** calibration finale. | Ne **pas** choisir le seuil 0,98 sur ce pilote ; validation indépendante indispensable. |
| **40 — Reproductibilité** | Seeds **42, 123, 2026** ; F1 = **1,000** sur test synthétique de 316 emails ; F1 original exploratoire moyen ≈ **0,9778** | Les performances synthétiques sont stables, mais les gabarits simplifient la tâche. | Sauvegardes vérifiées, configuration et historique incomplet correctement documentés. |
| **41 — Évaluation externe des 3 seeds** | F1 moyen **0,6977 ± 0,0213** ; rappel **0,9833 ± 0,0072** ; FPR **0,8375 ± 0,0976**, avec les **mêmes 160 emails** | Chute de spécificité récurrente sur le pilote ; aucune affirmation de robustesse générale. | Comparaisons avec les trois autres modèles sur la **même sélection**, et benchmark plus valide si possible. |

**Détail des attaques de l'expérience 41 :** substitutions lexicales applicables à 42 positifs, commentaire HTML à 80 et URL *defanging* à 13. Aucun passage détecté → non détecté observé sur les exemples positifs éligibles dans les trois seeds. **Ce résultat ne démontre pas la robustesse** : les attaques sont limitées et la préservation de l'intention n'a pas été certifiée. Le *defanging* peut rendre un lien inopérant et doit être traité comme **sonde de sensibilité**, pas nécessairement comme attaque d'évasion valide.

## 4. Protocole commun obligatoire avant toute comparaison

### 4.1 Trois statuts de données — ne pas mélanger

| Statut | Sources autorisées et usage | Peut servir de conclusion « phishing réel » ? |
|---|---|---|
| **A — Test synthétique séparé par gabarit** | `train.parquet` (1 307), `validation.parquet` (277), `test.parquet` (316), `original_exploratory.parquet` (100). L'entraînement/validation du modèle 40 utilise les partitions préexistantes. | **Non**, seulement robustesse et reproductibilité sur le domaine synthétique. Le test a déjà été examiné. |
| **B — Pilote externe commun exploratoire** | Manifeste privé Exp. 38 : 80 positifs Nazario, 80 négatifs CEAS ; `sample_id` pour joindre les prédictions. | **Non**, corpus et classes parfaitement confondus, labels hétérogènes, évaluation déjà inspectée. |
| **C — Sous-ensemble validé** | Nouvel ensemble privé de messages individuellement vérifiés `PHISHING/LEGITIMATE`, provenance contrôlée, sans recouvrement exact/proche avec le train, figé avant la comparaison. | **Oui, avec prudence**, selon taille, qualité des annotations et indépendance ; n'appeler « untouched test » **que** s'il n'a réellement jamais été consulté. |

**Décision minimale pour le 24 novembre :** les collègues peuvent comparer les méthodes sur **B**, en titrant les tableaux **« pilote exploratoire à étiquettes de corpus »**. En parallèle, le responsable benchmark prépare C. Si C n'est pas prêt, **aucune performance prétendument finale sur phishing réel** ne sera inventée.

**Vigilance fuite de données :** si TF-IDF / URL sont entraînés sur CEAS, ils **ne doivent pas** être testés sur les mêmes emails CEAS du manifeste B. Deux solutions : (i) entraîner TF-IDF/URL sur le même train/validation synthétique que DistilBERT pour une comparaison contrôlée **avec les mêmes limitations** ; ou (ii) construire un jeu d'évaluation disjoint documenté, en vérifiant doublons exacts et voisins très similaires. **Ne jamais** optimiser un modèle sur les 160 emails puis rapporter son score sur eux comme un vrai test.

### 4.2 Format de l'évaluation partagé à tous

Chaque modèle fournit **une prédiction par `sample_id`, condition et seed**, de préférence aussi le score (si disponible) ; dans la sortie publique, ne garder ni texte ni URL.

```csv
model,model_version,seed,sample_id,condition,source,label,pred,score,eligible,text_changed,latency_ms
TFIDF,tfidf_v1,42,opaque_id,clean,CEAS_08,0,0,0.123,false,false,1.2
```

- `condition` ∈ `{clean, lexical_proxy, neutral_hidden_html, url_defang}` **pour retrouver les pilotes historiques**. Toute **nouvelle** transformation doit avoir une version, un nom différent et un contrôle de validité.
- `score` peut être vide pour le LLM non probabiliste ; `seed` peut être `not_applicable` pour un LLM à configuration fixe, sans feindre trois entraînements.
- `label` = label du corpus **avec son statut**, pas « vérité terrain vérifiée » par défaut.
- `eligible` = la transformation est applicable ; `text_changed` = elle a effectivement changé le texte. Ne pas utiliser `eligible` pour les négatifs quand l'attaque n'est définie que pour les positifs.
- **Ne jamais** pousser `pilot_manifest.csv`, corps d'emails, prompts contenant des emails ni liens réels dans GitHub. Les prédictions détaillées avec IDs sont à auditer et à partager seulement en privé si elles permettent une jointure aux sources.

### 4.3 Métriques / définitions non négociables

- **TP, FP, TN, FN** : classificateur binaire, classe positive = phishing ; afficher la matrice de confusion.
- **F1 phishing** = `2 × précision × rappel / (précision + rappel)` ; calculé sur l'ensemble complet à chaque condition.
- **FPR légitime** = `FP/(FP+TN)` ; ne jamais remplacer par `1-accuracy`.
- **ΔF1** = `F1_attaqué - F1_propre` à seuil figé ; indiquer le signe.
- **ASR d'évasion** = `# positifs correctement détectés en propre puis manqués après attaque valide / # positifs correctement détectés en propre, éligibles et attaqués validement` ; fournir le **dénominateur**. Les « aucun flip » sur 13 URL ne certifient pas un ASR nul général.
- **Couverture** = `# éligibles/# positifs` et `# effectivement modifiés/# positifs` ; essentiel pour URL (13/80 dans Exp. 41).
- **Coût** : moyenne/ médiane/ p95 de temps/email, distinguer tokenisation, inférence et chargement ; enregistrer matériel et batch size ; pour API compter tokens et coût réel documenté.
- **Incertitude** : moyenne ± écart-type des 3 seeds **sur les métriques du modèle**, IC de Wilson pour proportions, bootstrap **apparié par message** pour différences de F1 ; test de McNemar si pertinent pour prédictions binaires appariées. **160 emails répétés sur 3 seeds ≠ 480 observations indépendantes.**

### 4.4 Politique d'attaque et validité

| Attaque | Mécanisme testé | Condition de validité | Risque de mauvaise interprétation |
|---|---|---|---|
| **Paraphrase / lexical** | Remplacer mots et formulations avec intention frauduleuse conservée | Deux vérifications humaines ou contrôle explicite de l'intention, message complet ; aucune suppression de la demande frauduleuse | Un remplacement très faible qui ne change pas la décision n'établit pas une robustesse large. |
| **Texte caché HTML** | Insertion de contenu peu/pas visible dans un client email, mais visible au modèle s'il lit le HTML brut | Définir si l'entrée est texte brut, HTML brut ou texte *rendu* ; vérifier que le message affiché reste identique | Sinon, les modèles ne reçoivent pas la même information. |
| **Obfuscation URL** | Changement de représentation du lien pour tromper des caractéristiques URL | Tester **hors ligne** sur des domaines de démonstration inertes ; prouver que l'action malveillante simulée reste représentable et que l'attaque n'est pas un simple lien cassé | Transformer `https://` en `hxxps://` peut neutraliser le lien ; c'est un diagnostic, pas nécessairement une évasion exploitable. |

Pour l'étude comparative, **geler d'abord les modifications** dans un manifeste privé commun. Les collègues ne doivent pas fabriquer chacun leurs propres attaques pour ensuite comparer des ASR incomparables. Les modèles LLM ne doivent pas recevoir le vrai label ni une explication du caractère malveillant dans le prompt d'inférence.

## 5. Répartition exacte du travail et critères d'acceptation

### Wijdane — Responsable des travaux déjà réalisés (TF-IDF, NB, MiniLM, DistilBERT), intégration et rédaction scientifique

**À faire maintenant (sans réentraîner) :**

1. Vérifier les exports de l'Exp. 40/41 et conserver les trois checkpoints dans Drive ; identifier version du code, modèle de base, matériel (Tesla T4), seeds (42/123/2026), seuil et tokenisation (256 tokens).
2. Rédiger **2–3 pages de travail** « Méthodologie des baselines et DistilBERT, résultats, robustesse, limites » ; réutiliser les métriques historiques et ne pas comparer directement des corpus différents ; deux tableaux obligatoires : (A) synthétique vs externe ; (B) trois seeds + attaques + couverture.
3. Conserver **0,50** comme seuil de l'Exp. 41 et citer les seuils de l'Exp. 39 uniquement dans la section « sensibilité exploratoire ».
4. Expliquer noir sur blanc les étiquettes CEAS spam/ham, l'absence de validation indépendante des 8 suspects, l'assistance IA à l'annotation, l'inspection antérieure des jeux externes, la troncature, et la confidentialité.
5. Publier uniquement README, protocole, scripts, tableaux agrégés anonymisés ; vérifier que GitHub **public** ne reçoit ni message ni checkpoint.
6. Contrôler les sorties des trois collègues (nombres de lignes, IDs identiques, labels, définition du seuil, pas de fuite). Consolider les tableaux et rédiger la conclusion qui répond aux RQ.

**Livrables :** `docs/DISTILBERT_METHOD_RESULTS.md`, tableau figures propre, `docs/CONSOLIDATED_FINDINGS.md`, reproduction README.
**Accepté si :** statistiques identiques aux exports 40/41 ; toute limite pertinente visible ; aucune affirmation « robuste » basée sur 0 flip dans une attaque faible.

### Collègue 1 — Détecteur LLM génératif (zéro-shot)

**But scientifique :** déterminer si l'inférence contextualisée avec des consignes figées fournit un avantage sur TF-IDF/URL/DistilBERT **sur les mêmes messages**.

**Étapes obligatoires :**

1. Choisir **un seul modèle documenté** (ex. Ollama local ; nom, famille, version, quantification, taille, licence). Vérifier que le matériel peut le faire tourner et que les données ne partent pas vers un service non autorisé.
2. Concevoir un **prompt unique**, figé avant les tests, qui définit PHISHING vs LEGITIMATE, traite le mail comme des données non fiables, ignore les instructions contenus *dans l'email*, et rend uniquement un JSON `{"label":"PHISHING|LEGITIMATE"}`. Pas d'exemples de test dans le prompt.
3. Fixer température (idéalement 0) et paramètres (`num_ctx`, `num_predict`, etc.) ; capturer la version et le hash/identifiant du modèle si accessible.
4. Tester quelques **exemples de validation non réservés à l'évaluation** : vérifier parse JSON, taux de réponse refusée/invalide ; choisir une règle explicite de traitement des erreurs. **Geler le prompt**.
5. Exécuter exactement les **mêmes 160 `sample_id`** sur les conditions propres et modifiées ; **ne pas** changer le prompt entre conditions. Capturer prédiction, éventuel refus/parse error, temps par email, tokens générés/entrée si disponibles.
6. Produire matrice de confusion, F1/recall/FPR et ΔF1/ASR conditionnel ; documenter la forte limite du pilote. Ne pas comparer directement des latences sur matériels différents sans le signaler.
7. Analyser **5 FP et 5 FN** représentatifs (redigés sans reproduire d'emails privés) ; vérifier les cas de prompt injection et de sortie invalide.

**Livrables :** `src/models/llm_detector.py`, `configs/llm_prompt_v1.txt`, `results/metrics/llm_aggregate.csv` (anonymisé), `docs/LLM_REPORT.md`, instructions d'installation Ollama.
**Accepté si :** prompt et paramètres figés, 4 conditions, IDs cohérents, erreurs comptabilisées, pas d'envoi externe non autorisé, aucune confusion « LLM qui écrit du phishing » vs « LLM qui le détecte ».

### Collègue 2 — Nouveau modèle URL-only et réutilisation du TF-IDF existant

**But scientifique :** fournir le comparateur sur variables URL exigé et harmoniser les résultats des modèles déjà réalisés. **Ne pas réimplémenter TF-IDF.**

1. Lire `src/models/train_baseline.py`, `train_group_baseline.py` et surtout `train_phishing_templates.py`, ainsi que `results/metrics/phishing_template_baseline.json`. Wijdane a déjà entraîné le TF-IDF synthétique sur 1 307 messages, choisi le seuil 0,44 sur validation et obtenu F1 0,8858 sur 316 messages.
2. Charger le checkpoint privé TF-IDF correspondant au train synthétique, si disponible. Sinon relancer le script existant avec ses paramètres d'origine, sans changer les splits. Contrôler recouvrement exact/proche avec le manifeste, et conserver 0,44 pour la reproduction historique. Pour une nouvelle comparaison à FPR comparable, définir une règle de sélection sur validation indépendante avant de voir le nouveau test ; ne pas ajuster le seuil sur le pilote 38/41. Le TF-IDF CEAS ne doit pas être évalué comme indépendant sur le pilote contenant des CEAS train/validation.
3. Implémenter le **nouveau** modèle URL-only sur variables lexicales et structurelles extraites hors ligne ; aucun accès réseau aux liens ni exécution. Consigner les règles de traitement des messages sans URL, la couverture et les champs manquants. L'indicateur binaire CEAS `urls` n'est pas une liste de liens. Si le train synthétique ne contient pas assez de liens ou de variation, déclarer cette insuffisance et choisir un train/validation approprié documenté, sans mélange avec le test.
4. Entraîner une baseline simple sur train, régler le seuil sur validation ; même benchmark et variantes figées que les autres détecteurs. Pour tout nouveau modèle stochastique : graines 42, 123, 2026, moyenne ± écart-type ; pour un algorithme déterministe, documenter l'absence de variation et la stratégie de répétition, sans inventer des entraînements indépendants.
5. Évaluer propre/perturbé sur les IDs et variantes communs ; conserver les transformations historiques comme sondes exploratoires. Tester le sous-ensemble URL éligible et l'ensemble complet ; rapporter F1, FPR, ASR conditionnel, couverture et coût/message. Pour le nouveau benchmark, attendre la validation et le gel par le collègue 3 avant évaluation.
6. Ablation : retirer des familles de variables du nouveau modèle URL. Comparer URL-only vs TF-IDF existant ; fusion texte+URL facultative. Les ablations historiques de Wijdane restent disponibles et ne doivent pas être répétées sans raison.

**Livrables :** `src/models/train_url_features.py`, adaptateur `src/evaluation/eval_classical_exp42.py`, config/checkpoint privé, métriques agrégées, `docs/CLASSICAL_RESULTS.md` et preuve d'absence de fuite. **Accepté si :** nouveau URL-only fonctionnel, baseline TF-IDF réutilisée, mêmes variantes, règles de seuil documentées, aucune fuite, coût et couverture mesurés. Aucun nouveau code d'entraînement TF-IDF redondant n'est requis.

### Collègue 3 — Benchmark, qualité des labels, attaques et statistiques

**But scientifique :** rendre la réponse à la question de recherche **défendable** ; une comparaison très précise sur des étiquettes erronées est inutile.

**Étapes obligatoires :**

1. Rédiger un **dictionnaire de labels** : PHISHING = tentative démontrée de tromper pour obtenir information, paiement ou action frauduleuse ; SPAM = sollicitation de masse sans preuve de ce but ; LEGITIMATE = correspondance non frauduleuse ; UNCERTAIN = preuves insuffisantes.
2. Faire une **annotation humaine indépendante** d'un échantillon privé ciblé ; ne pas consulter les suggestions de ChatGPT ou du second LLM avant de coder les jugements. Documenter quels messages ont déjà reçu une aide IA.
3. Résoudre les **8 cas CEAS contestés**, dont le message de mise à jour de paiement ; une fausse identité de type newsletter ne prouve pas à elle seule le phishing. Préserver les jugements bruts et arbitrages.
4. Vérifier exact hashes/proximité/« même gabarit » entre train, validation et candidats test. Documenter le caractère **déjà consulté** de CEAS test et SpaPhish. Vérifier aussi couverture URL par source et classe.
5. Construire **un manifeste privé figé** : version, `sample_id`, source, partition, label, statut de validation, type d'attaque, indicateur `eligible`, trace de validité sémantique. Publier seulement agrégats.
6. Auditer les attaques : sens maintenu, forme HTML rendue/brute et fonctionnalité du lien d'essai. Comparer les modèles **sur les mêmes textes et conditions**.
7. Calculer comparaison appariée, IC Wilson, bootstrap par email et éviter de multiplier la taille d'échantillon par 3 pour les seeds. En cas de données insuffisantes, reporter des IC larges et aucune conclusion forte.
8. Consolider 3–5 exemples d'erreurs **anonymisés** et leurs causes plausibles, sans prétendre à la causalité prouvée.

**Livrables :** `docs/BENCHMARK_CARD.md`, `docs/ANNOTATION_PROTOCOL.md`, `docs/ATTACK_VALIDITY.md`, `results/metrics/benchmark_aggregate.csv`, `docs/STATISTICAL_COMPARISON.md`.
**Accepté si :** aucun chevauchement non justifié, étiquettes et preuves documentées, couverture d'attaque visible, comparaison appariée, aucune donnée privée publiée.

### Collaboration — contrat d'intégration

Chaque collègue remet **un commit/PR** + (i) commande exécutable depuis la racine du repo, (ii) config/seeds, (iii) tableau agrégé CSV, (iv) rapport 1 page avec **1 constat, 1 limite, 1 exemple d'erreur**, (v) journal des dépendances, (vi) confirmation qu'aucun email réel n'a été publié. **Les résultats doivent être reproductibles sans accéder au PC de Wijdane**, dès que les détenteurs des données ont approuvé leur accès privé.

## 6. Calendrier réaliste et dépendances

| Date limite | Responsable | Tâches prioritaires / livrable vérifiable | Dépend de |
|---|---|---|---|
| **21 novembre (maintenant)** | Wijdane | Publier roadmap ; transmettre protocole et format de métriques ; préserver les exports 38–41 ; créer les tâches GitHub. | Aucune |
| **22 novembre — matin** | Collègue 3 | Manifeste commun figé pour comparaison **exploratoire** + labels statuts ; premiers contrôles de fuite. | Accès privé autorisé aux textes |
| **22 novembre — soir** | Collègues 1 et 2 | Prototype LLM / URL et adaptateur de comparaison TF-IDF existant ; validation locale sur un jeu non-test ; configuration figée. | Protocole et fichiers train/val |
| **23 novembre — midi** | Collègues 1 et 2 | Prédictions clean et 3 attaques ; matrices de confusion, F1/FPR/latence, ablations. | Manifeste et transformations figées |
| **23 novembre — soir** | Collègue 3 + Wijdane | Comparaison appariée, vérification qualité des attaques, synthèse de l'échec de généralisation et limites. | Prédictions des 4 modèles |
| **24 novembre** | Toute l'équipe | **Jalon expérimental :** tableaux complets, figures, ablations, erreurs, métriques et protocole documenté ; marquer explicitement toute lacune restante. | Intégration |
| **25–29 novembre** | Toute l'équipe | Renforcer labels/benchmark si possible ; lire les articles prioritaires, écrire état de l'art, méthodes et risques. | Jalon figé |
| **30 novembre–4 décembre** | Wijdane + contributeurs | Article 8–12 p. **2 colonnes**, figures, bibliographie 15–25 sources, relecture croisée ; diaporama 12–15 slides. | Résultats et état de l'art |
| **5–7 décembre** | Toute l'équipe | Tests de reproduction, cohérence texte/tableaux, citations, transparence IA, répétition orale. | Article et code |
| **8 décembre** | Toute l'équipe | **Remise finale :** article PDF et Word, PPTX et dépôt reproductible. | Validation finale |

**Chemin critique :** benchmark commun → modèles classiques et LLM sur les mêmes données → agrégation statistique → article. La rédaction et la lecture commencent **en parallèle**, sans attendre la dernière exécution.

## 7. Plan de l'article (8–12 pages, deux colonnes) et des 12–15 slides

| Article | Volume indicatif | Figure/table exigée |
|---|---|---|
| Résumé + mots-clés + introduction | 0,8–1,2 p. | Question de recherche et contributions sans survente |
| État de l'art / lacunes | 1,3–1,8 p. | Tableau « études précédentes vs notre protocole » |
| Données, annotations, menaces | 1,2–1,6 p. | Composition des corpus, risques de labels/provenance |
| Modèles et réglages | 1,2–1,5 p. | Tableau TF-IDF / URL / DistilBERT / LLM |
| Attaques, validité et métriques | 1–1,4 p. | Pipeline propre → transformation → détection ; définitions |
| Résultats, ablations et statistiques | 1,8–2,5 p. | Table clean/attack, FPR, ASR, couverture, temps, ± SD |
| Discussion, erreurs, limites, éthique | 1–1,5 p. | Différence synthétique/externe et limites causales |
| Conclusion + bibliographie | 1–1,5 p. | Réponse nuancée à la question |

**Diapositives proposées (14) :** 1 titre ; 2 question et hypothèses ; 3 menaces/adversaire ; 4 corpus/qualité ; 5 protocoles et fuite ; 6 quatre modèles ; 7 attaques et validité ; 8 métriques ; 9 résultats synthétiques ; 10 trois seeds et généralisation ; 11 comparaison tous modèles ; 12 ablations/erreurs ; 13 limites ; 14 conclusion + perspectives.

## 8. Bibliographie guidée — liste de départ à compléter

**Exigence du professeur : 15 à 25 références, dont une majorité de moins de trois ans.** La liste de départ ci-dessous privilégie des fondements anciens et ne remplit pas encore cette contrainte. Pour 20 références finales, au moins 11 doivent dater de moins de trois ans à la remise ; remplacer une partie des références anciennes par des publications pertinentes 2024–2026, vérifier métadonnées, DOI, date et contenu auprès des éditeurs, puis sélectionner celles réellement utilisées. Ne pas considérer les 20 entrées comme bibliographie finale vérifiée à nouveau dans cette révision.

Ce ne sont pas vingt citations à coller sans lecture : ce sont **vingt références candidates** et directement liées à un choix méthodologique. Dans le papier, citer uniquement ce qui a été lu et réellement utilisé. La **priorité P0** signifie lecture immédiate ; **P1** avant rédaction des méthodes ; **P2** en complément/discussion. Pour chaque papier, remplir une fiche : *question, données, étiquettes, split, attaques, métriques, limites, ce que nous reprenons/ne reprenons pas*.

| # / Priorité | Référence et lien officiel | Pourquoi la lire et ce qu'il faut vérifier |
|---|---|---|
| **1 · P0** | [E-PhishGen: Unlocking Novel Research in Phishing Email Detection (AISec 2025)](https://doi.org/10.1145/3733799.3762967) | **Collègue 3 + Wijdane.** Reproductibilité, données phishing/legit, risques des benchmarks historiques ; comparer notre transparence des labels et des sources. |
| **2 · P0** | [PhishBench 2.0 (ACM CCS 2020)](https://doi.org/10.1145/3372297.3420017) | **Tous.** Justifie des datasets, métriques et conditions de comparaison unifiés ; identifier ce qu'est une comparaison équitable. |
| **3 · P0** | [An In-Depth Benchmarking and Evaluation of Phishing Detection Research for Security Needs (IEEE Access 2020)](https://doi.org/10.1109/ACCESS.2020.2969780) | **Collègue 2 + 3.** Examiner sélection des caractéristiques, normalisation des expériences et validité du benchmark. |
| **4 · P0** | [Is BERT Really Robust? / TextFooler (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/6311) | **Collègue 3.** Contraintes de substitutions, sémantique et attaque ; comparer aux proxies faibles d'Exp. 41. |
| **5 · P0** | [TextAttack: A Framework for Adversarial Attacks (EMNLP 2020)](https://aclanthology.org/2020.emnlp-demos.16/) | **Collègue 3.** Formaliser transformations, contraintes, objectif de l'attaquant, taux de réussite et reproductibilité. |
| **6 · P0** | [DistilBERT, a distilled version of BERT (2019)](https://arxiv.org/abs/1910.01108) | **Wijdane.** Justifier distillation, compromis taille/vitesse et limites d'un encodeur à 256 tokens. |
| **7 · P0** | [On Calibration of Modern Neural Networks (ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html) | **Wijdane + 3.** Distinguer score softmax et probabilité calibrée ; choisir seuil sur validation, non sur Exp. 38 inspecté. |
| **8 · P0** | [WILDS: A Benchmark of in-the-Wild Distribution Shifts (ICML 2021)](https://proceedings.mlr.press/v139/koh21a.html) | **Wijdane + 3.** Lire les risques du changement de distribution et expliquer le gouffre synthétique/externe sans le sur-interpréter. |
| **9 · P1** | [BERT: Pre-training of Deep Bidirectional Transformers (NAACL 2019)](https://aclanthology.org/N19-1423/) | **Wijdane.** Fondement du modèle de langage contextualisé et du fine-tuning de classification. |
| **10 · P1** | [BERT-Attack (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.500/) | **Collègue 3.** Comprendre attaques par masque MLM et leurs contraintes ; **ne pas** intégrer sans validation de l'intention frauduleuse. |
| **11 · P1** | [HotFlip: White-Box Adversarial Examples for Text Classification (ACL 2018)](https://aclanthology.org/P18-2006/) | **Collègue 3.** Pourquoi les perturbations caractère/typos peuvent défavoriser les modèles lexicaux. |
| **12 · P1** | [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList (ACL 2020)](https://aclanthology.org/2020.acl-main.442/) | **Tous.** Construire des tests par familles comportementales au lieu de conclure sur un seul F1. |
| **13 · P1** | [Beyond Blacklists: Learning to Detect Malicious Web Sites from Suspicious URLs (KDD 2009)](https://cseweb.ucsd.edu/~jtma/papers/beyondbl-kdd2009.pdf) | **Collègue 2.** Justifier variables lexicales/structurelles URL ; préciser que l'étude traite de sites malveillants, pas directement de phishing email. |
| **14 · P1** | [URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection (2018)](https://arxiv.org/abs/1802.03162) | **Collègue 2.** Comprendre signaux caractères/mots d'URL et ablation URL-only ; ne pas confondre URLNet et notre baseline simple. |
| **15 · P1** | [In-Depth Analysis of Phishing Email Detection Across Multiple Datasets (Applied Sciences 2025)](https://www.mdpi.com/2076-3417/15/6/3396) | **Collègue 2 + 3.** Examiner comparaisons inter-corpus et avertissements sur composition des jeux. |
| **16 · P1** | [Spanish phishing and legitimate email dataset with technical and psychological annotations (Data in Brief 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13273897/) | **Collègue 3.** Comprendre la provenance/annotation SpaPhish, les deux classes et le défi linguistique ; noter que nous avons **déjà inspecté** ce corpus. |
| **17 · P1** | [APOLLO: A GPT-based tool to detect phishing emails and generate explanations (ACM 2025)](https://doi.org/10.1145/3733049) | **Collègue 1.** Définition de la tâche LLM détecteur et évaluation d'explications ; distinguer détection et génération. |
| **18 · P1** | [Evaluating LLMs Versus ML in Detecting Phishing Email Attempts (Algorithms 2025)](https://www.mdpi.com/1999-4893/18/10/599) | **Collègue 1 + 2.** Examiner protocole, prompts, emails/URL et métriques ; observer les éventuels facteurs de confusion. |
| **19 · P2** | [A Two-Stage Deep Learning Framework for AI-Driven Phishing Email Detection Based on Persuasion Principles (Computers 2025)](https://doi.org/10.3390/computers14120523) | **Wijdane + 1.** Extraire l'usage de DistilBERT et des indices de persuasion ; discuter les signaux comportementaux, **sans** recréer l'architecture. |
| **20 · P2** | [Cross-model evaluation of phishing detectors against LLM-generated emails (Frontiers 2026)](https://doi.org/10.3389/fdata.2026.1883452) | **Collègue 1 + 3.** Lire sur généralisation et calibration ; **attention** : l'étude cible principalement la distinction *phishing humain vs généré par LLM*, et **non** *phishing vs légitime*. Ne pas utiliser ses F1 comme comparaison directe. |

**Lecture complémentaire directement concurrente, facultative mais utile pour la discussion :** [Adversarial Robustness of Phishing Email Detection: TF-IDF + Logistic Regression vs DistilBERT (prépublication, juillet 2026)](https://arxiv.org/abs/2607.18429). Comparer les jeux de données, attaques, protocole d'étiquetage et contrôles de fuite avant d'emprunter ses conclusions. Un article de *preprint* n'est pas automatiquement évalué par les pairs.

### Matrice de lecture à renseigner pour chaque papier (copier une fois par source)

```
Citation / DOI :
Problème et hypothèse :
Corpus et provenance des étiquettes :
Train / validation / test et contrôle de fuite :
Modèles et seuils :
Attaques, contraintes et vérification de sens :
Métriques, dénominateurs, coûts et IC :
Conclusions réellement étayées :
Limites / risque de source ou de domaine :
Choix de CYB-5 justifié ou contredit :
Phrase du papier où citer cette source :
```

## 9. Contrôles avant publication GitHub et remise

1. **Confidentialité :** exécuter `git status --short`, `git ls-files data models`, inspecter fichiers déjà suivis ; vérifier notebooks pour outputs d'emails bruts, noms privés, tokens et URL. Les hashes déjà présents dans des CSV publics peuvent permettre des rapprochements ; évaluer leur pertinence et leur conservation.
2. **Reproductibilité :** README, versions dépendances adaptées à Windows/Colab, commandes de lancement, graine, modèle, matériel, chemins privés *paramétrables*, description de l'acquisition légale des données.
3. **Comparaison :** mêmes IDs et attaques, pas de fuite, statut des labels, scores figés sans optimisation sur test, couverture des transformations, FPR légitime, latence.
4. **Résultats :** distinguer synthétique, pilote exploratoire, évaluation vérifiée ; aucun « 100% sur phishing réel » si obtenu sur gabarits synthétiques.
5. **Statistiques :** trois seeds ≠ trois jeux indépendants ; intervalles et tests appariés correctement renseignés ; pas de fausse significativité avec peu d'URL.
6. **Article :** 8–12 pages, deux colonnes, 15–25 références réellement utilisées, majorité de moins de trois ans, captures pertinentes, limites, déclarations d'assistance IA et de l'usage de Colab / autres outils ; vérification finale de toutes les dates et valeurs.
7. **Plan de secours :** si benchmark C n'est pas prêt au 24 novembre, remettre B comme **étude exploratoire**, expliciter les travaux futurs, ne jamais maquiller la vérification des labels ou le statut du test.

## 10. Définition du succès pour le professeur

Nous pouvons répondre à la question centrale **seulement** si les modèles ont été mesurés sur les mêmes emails, avec **les mêmes attaques valides**, en affichant **F1 propre et attaqué, taux de faux positifs légitimes, couverture et latence**. Nous pouvons conclure « dans ce protocole et avec ces limites, le modèle X subit une baisse de F1 de Y tandis que la baseline perd Z » ; nous **ne pouvons pas** extrapoler « X est robuste en production » à partir d'un petit pilote confondu et de 0 flips sur des attaques simplifiées.

**Priorité opérationnelle :** benchmark commun privé et figé → réutilisation TF-IDF + nouveau URL-only + nouveau LLM → agrégation appariée → rédaction. Le travail DistilBERT est suffisamment avancé ; la valeur scientifique restante vient d'une **comparaison fiable**, pas d'un quatrième entraînement.


## Annexe — Preuves du travail déjà réalisé (audit du dépôt)

Commit vérifié : [b3f597852b5179f9a48207b829768ec7d931f16d](https://github.com/wijdane-ARB/CYB5/commit/b3f597852b5179f9a48207b829768ec7d931f16d).

- [Scripts d'entraînement classiques](https://github.com/wijdane-ARB/CYB5/tree/b3f5978/src/models) : TF-IDF + LR (CEAS, groupé, TREC, synthétique), ComplementNB.
- [Attaques](https://github.com/wijdane-ARB/CYB5/tree/b3f5978/src/attacks), [défenses](https://github.com/wijdane-ARB/CYB5/tree/b3f5978/src/defenses), [évaluation](https://github.com/wijdane-ARB/CYB5/tree/b3f5978/src/evaluation).
- [Métriques historiques](https://github.com/wijdane-ARB/CYB5/tree/b3f5978/results/metrics) et [notebooks exécutés](https://github.com/wijdane-ARB/CYB5/tree/b3f5978/notebooks).
- Les directives originales exigent aussi une commande unique de reproduction, un article Word + PDF, et un PPTX modifiable. Les historiques epoch des 3 checkpoints ne sont pas intégralement disponibles : le signaler.
- Le dépôt public contrôlé ne contient pas encore les scripts/mesures finales d'un LLM génératif ou d'un classificateur URL-only, ni de comparaison finale validée. Cela ne prouve pas qu'aucun collègue n'a commencé ces travaux ailleurs.
