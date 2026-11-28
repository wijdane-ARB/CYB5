# LLM zero-shot (Ollama) — installation et exécution

Détecteur local : **aucun email ne quitte la machine** (le code refuse tout point d'accès non-loopback). Les poids du modèle sont les seules données téléchargées.

## 1. Prérequis

- Python 3.11+ avec `requirements.txt` du dépôt (le détecteur n'utilise que la bibliothèque standard + `pandas`/`numpy`).
- Le fichier privé **original** `data/audit/cyb5_experiment38_pilot/pilot_manifest.csv` (160 messages, 80/80). Ne pas le régénérer : le script vérifie qu'il reproduit la couverture de l'Exp. 41 (42 / 80 / 13 positifs modifiés). **Ne jamais le versionner.**
- Un GPU est recommandé (Colab T4 suffit pour un modèle 7B quantifié Q4) ; le CPU fonctionne mais est lent.

## 2. Installer Ollama et le modèle

**Windows** : installer depuis <https://ollama.com/download>, puis dans CMD :

```bat
ollama serve
:: (autre fenêtre)
ollama pull qwen2.5:7b-instruct
```

**Linux / Colab** :

```bash
curl -fsSL https://ollama.com/install.sh | sh     # sur Colab : apt-get install -y zstd pciutils d'abord
ollama serve &
ollama pull qwen2.5:7b-instruct
```

Le modèle, ses paramètres et l'empreinte SHA-256 du prompt sont figés dans `configs/llm_config_v1.json`. Pour changer de modèle ou de prompt, créer `llm_config_v2.json` / `llm_prompt_v2.txt` ; ne pas modifier v1 après les tests.

## 3. Commandes de reproduction (depuis la racine du dépôt)

```bat
cd /d D:\CYB5
.venv\Scripts\activate
python -m unittest discover -s tests -v
python -m src.evaluation.llm_pilot_eval --mode validate
python -m src.evaluation.llm_pilot_eval --mode run
```

- `validate` : 14 messages **synthétiques inertes** (non liés au pilote) pour vérifier le parsing JSON, le taux de réponses invalides et deux cas d'injection. À faire **avant** `run`.
- `run` : 160 messages × 4 conditions (`clean`, `lexical_proxy`, `neutral_hidden_html`, `url_defang`), reprise automatique si interrompu (`--restart` pour repartir de zéro).

## 4. Sorties

| Fichier | Contenu | Versionnable ? |
|---|---|---|
| `data/audit/cyb5_llm_pilot/llm_predictions_PRIVATE.csv` | une ligne par `sample_id` × condition (format §4.2), **IDs mais pas de texte** | **Non** (privé) |
| `data/audit/cyb5_llm_pilot/llm_validation_report.json` | résumé de la validation synthétique | Non |
| `results/metrics/llm_aggregate.csv` | 4 lignes agrégées, sans ID ni texte | Oui, **après relecture** |
| `results/metrics/llm_run_record.json` | modèle + empreintes, paramètres, matériel, couverture ; sans chemins ni texte | Oui, après relecture |

## 5. Contrôle avant tout push

```bat
git status --short
git ls-files data
git diff --cached --stat
```

`git ls-files data` doit être vide. Ouvrir `llm_aggregate.csv` : aucune colonne `sample_id`, aucun texte, aucun lien.
