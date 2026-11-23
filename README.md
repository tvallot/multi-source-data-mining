# Multi-source data mining : retrouver une histoire cachée dans quatre jeux de données

> **TL;DR (EN)** — Group data-mining project (Master 2 AI / Data Science). Four synthetic datasets (public records, health, ecology, logistics) hide one coherent story. Each dataset is cleaned and analysed on its own (statistics, PCA, hierarchical clustering, DBSCAN, association rules, time-series decomposition, placebo tests), then cross-linked in a final synthesis notebook. Notebooks are written in French.

## Le projet

Quatre jeux de données fictifs, issus de domaines différents (registres publics, santé, écologie, logistique), n'ont aucun lien apparent. Le but est de **retrouver l'histoire qu'ils racontent ensemble**, uniquement à partir des données et avec les méthodes vues en cours.

Chaque domaine est nettoyé et analysé séparément, puis un notebook de synthèse relie les résultats et vérifie les liens directement dans les données.

## Ce que l'analyse met en évidence

| Constat | Domaines | Preuve |
|---|---|---|
| Après **chacun** des 14 accès nocturnes d'une même organisation à des sites naturels, le nombre de chauves-souris chute d'environ **59 %** | registres publics + écologie | comparaison à la normale de saison, test placebo (p ≈ 0,0005) |
| **18 envois sur 18** de colis réfrigérés partent exactement 6 h après la fin d'un de ces accès | registres publics + logistique | croisement des dates et des sites |
| Des lots biologiques **disparaissent des stocks** des entrepôts qui reçoivent ces colis | logistique | écarts d'inventaire, ajustements de stock |
| **10 patients** ont des signaux physiologiques incompatibles avec la norme humaine (deux rythmes cardiaques séparés, démarche atypique) | santé | PCA, classification hiérarchique, DBSCAN, analyse des ECG |
| **8 dirigeants** ont un âge biologique inférieur d'environ 13 ans à leur âge réel ; **30 injections sur 30** correspondent à un retrait de stock le même jour | santé + logistique | comparaison âge biologique / âge réel, croisement des tables |

Le détail de l'enquête, étape par étape, est dans [`notebooks/00_synthese.ipynb`](notebooks/00_synthese.ipynb). Les données étant synthétiques, l'histoire est entièrement fictive.

## Méthodes utilisées

- **Nettoyage et qualité des données** : détection de doublons, valeurs aberrantes marquées plutôt que supprimées, journal de nettoyage
- **Statistique descriptive** : distributions, corrélations de Pearson et de Spearman
- **Réduction de dimension et clustering** : PCA, classification hiérarchique, DBSCAN
- **Séries temporelles** : décomposition STL, saisonnalité, autocorrélation
- **Règles d'association** et analyse de processus
- **Inférence causale prudente** : tests placebo, vérification des variables confondantes (météo, appareils de mesure, autres espèces)

## Organisation du dépôt

```
├── data/                    Données fournies (brutes) et données nettoyées par les notebooks
│   ├── public_records/
│   ├── ecology/
│   ├── logistics/           (+ clean/ produit par le notebook de nettoyage)
│   ├── health/
│   ├── health_clean/        produit par le notebook de nettoyage santé
│   └── data_dictionary.csv
├── notebooks/
│   ├── 00_synthese.ipynb               Synthèse : relie les quatre domaines
│   ├── public_records.ipynb            Registres publics
│   ├── ecology.ipynb                   Écologie
│   ├── 01_nettoyage_logistics.ipynb    Logistique : nettoyage
│   ├── 02_analyse_logistics.ipynb      Logistique : analyse
│   ├── 01_nettoyage_health.ipynb       Santé : nettoyage
│   ├── 02_analyse_health.ipynb         Santé : analyse complète
│   ├── 03_rapport_health.ipynb         Santé : rapport court
│   └── health_utils.py                 Fonctions partagées du dossier santé
└── requirements.txt
```

**Par où commencer ?** Par `00_synthese.ipynb` pour le résultat global, puis `03_rapport_health.ipynb` pour un exemple de rapport complet sur un domaine. Les notebooks gardent leurs résultats affichés : il n'est pas nécessaire de tout relancer pour les lire.

## Lancer les notebooks

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Les notebooks doivent être lancés depuis le dossier `notebooks/` (ils lisent `../data/`). Lancer `01_nettoyage_*` avant les analyses correspondantes : ils créent `data/logistics/clean/` et `data/health_clean/`.

## Usage de l'IA générative

Chaque notebook précise comment l'IA générative a été utilisée pour sa partie.

## Crédits et licence

Sujet et jeux de données fournis dans le cadre de l'UE Data Mining (Master 2, 2026-2027) par Antoine T. Richard. Licence Beer-Ware, voir [`LICENSE`](LICENSE).
