# Prédiction de la classe de température à 2 m — Algérie

Projet de Data Mining (USTHB / FI / IASD — M2 SII 2026-2027).

Chaîne complète sur les réanalyses **ERA5-Land, moyennes mensuelles**, Algérie,
2015 → dernière année complète : analyse et prétraitement, classification
supervisée de la température à 2 m du mois suivant, puis clustering année par
année.

## Structure

```
data/raw/         NetCDF téléchargé du CDS
data/processed/   tables nettoyées et agrégées (parquet)
notebooks/        01_analyse_pretraitement · 02_supervise · 03_clustering
src/              config · download · preprocessing · supervised · clustering
reports/figures/  figures exportées
```

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Données

Zone, période, variables et résolution sont dans `src/config.py`.

Avec une clé Copernicus CDS dans `~/.cdsapirc` :

```bash
python -m src.download
```

Sans clé, un jeu de substitution de **structure identique** (même grille 0.1°,
mêmes 12 variables, mêmes unités, NaN hors terres, saisonnalité et tendance de
réchauffement) permet de faire tourner toute la chaîne :

```bash
python -m src.download --synthetic
```

## Chaîne complète

```bash
python -m src.download --synthetic   # ou sans l'option, avec une clé CDS
python -m src.preprocessing
python -m src.supervised
python -m src.clustering
```

---

## Étape 1 — Analyse et prétraitement

`src/preprocessing.py` · `notebooks/01_analyse_pretraitement.ipynb`

**Réduction spatiale.** `coarsen(latitude=2, longitude=2).mean()` agrège la
grille par blocs 2×2 (4 pixels) : 0.1° → 0.2°. `boundary="trim"` écarte les lignes et
colonnes en trop plutôt que de former un bloc partiel. La moyenne de xarray
ignore les NaN : un bloc à cheval sur la côte garde la moyenne de ses points
terrestres, un bloc entièrement en mer reste NaN.

**Valeurs manquantes.** `missing_report` classe chaque point de grille selon son
taux de NaN sur `t2m` : `terre`, `partiel`, ou `hors_terre` (> 99 % de NaN — mer
ou hors domaine ERA5-Land). `drop_sea_points` écarte les points hors terre et
interpole les trous isolés le long de la série de chaque point.

**Unités.** Toutes les conversions sont idempotentes : relancer le script ne
convertit pas deux fois.

| variables | de | vers |
|---|---|---|
| `t2m`, `d2m` | K | °C |
| `slhf`, `ssr`, `str`, `sshf` | J m⁻² (cumul journalier) | W m⁻² |
| `ro`, `e`, `tp` | m | mm/jour |
| `sp` | Pa | hPa |

**Exploration.** Distributions des 12 variables, cycle saisonnier, matrice de
corrélation, tendance de la moyenne annuelle.

Sortie : `data/processed/era5_clean.parquet`.

## Étape 2 — Apprentissage supervisé

`src/supervised.py` · `notebooks/02_supervise.ipynb`

**Instance** : les 11 variables explicatives du sujet (`d2m`, `slhf`, `ssr`,
`str`, `sshf`, `ro`, `e`, `u10`, `v10`, `sp`, `tp`) + `year`, `month` et
l'encodage cyclique du mois (`month_sin`, `month_cos`).

**Cible** : `t2m` du **mois suivant**, décalée à l'intérieur de chaque point de
grille. Un contrôle sur les dates invalide la cible quand les deux mois ne se
suivent pas réellement.

**Discrétisation** : 3 intervalles de **même largeur** (`np.linspace` sur
l'étendue — équi-largeur, pas équi-effectif), étiquetés `froid` / `tempere` /
`chaud`.

**Split chronologique** : les `TEST_YEARS` dernières années en test, aucun
mélange aléatoire.

> **Point de méthode.** Les bornes des intervalles sont calculées sur le **train
> seul** puis appliquées au test, avec des bornes extrêmes ouvertes (±∞) pour
> absorber les valeurs de test hors étendue. Les calculer sur l'ensemble complet
> ferait fuir dans le train une information sur l'étendue des températures
> futures.

**Modèles** : classe majoritaire (référence), KNN, SVM (RBF), forêt aléatoire,
AdaBoost, vote dur et vote souple sur KNN + SVM + forêt. KNN et SVM sont
encapsulés dans un pipeline avec `StandardScaler` — tous deux raisonnent sur des
distances.

> **Sous-échantillonnage.** `SVC` est quadratique en nombre de lignes. Par
> défaut tous les modèles sont entraînés sur le même sous-échantillon de
> `SVM_MAX_TRAIN` lignes du train, pour garder la comparaison à effectif égal ;
> la taille utilisée est affichée à chaque exécution. `evaluate(..., sample_size=None)`
> entraîne sur la totalité du train.

**Métriques** : accuracy, accuracy équilibrée, F1 macro et pondéré, kappa de
Cohen, rapport de classification et matrice de confusion.

Sortie : `data/processed/supervised.parquet` (colonne `split`).

## Étape 3 — Clustering

`src/clustering.py` · `notebooks/03_clustering.ipynb`

**Instance** : un couple (année, point de grille). Les couples incomplets (moins
de 12 mois) sont écartés.

| groupe | colonnes |
|---|---|
| valeurs mensuelles | `t2m_01`…`t2m_12`, `str_01`…`str_12` |
| tendance intra-annuelle | `_mean`, `_std`, `_amplitude`, `_pente` (pente OLS sur les 12 mois) |
| tendance long terme | `_anomalie` (écart à la normale du point, calculée sur les `BASELINE_YEARS` premières années), `_derive` (pente des moyennes annuelles du point sur toute la période) |

**Algorithmes**, appliqués **année par année** : K-Means, AGNES (agglomératif,
lien de Ward), DBSCAN, et AutoEncoder + K-Means. L'auto-encodeur est un
`MLPRegressor` à goulot `(32, b, 32)` entraîné à reconstruire son entrée ; le
K-Means tourne ensuite sur la couche goulot.

Deux choix nécessaires à l'analyse d'évolution :

- la **standardisation est globale**, pas par année, sinon les clusters d'années
  différentes ne sont pas comparables ;
- les **étiquettes sont renumérotées** par température moyenne croissante
  (`_canonical`), sinon elles sont arbitraires à chaque exécution et la matrice
  de transition n'a aucun sens.

**Choix de `eps` pour DBSCAN** : balayage des centiles de la courbe des
k-distances, en retenant le `eps` qui maximise la silhouette sous contrainte de
bruit (`DBSCAN_MAX_NOISE`) ; repli sur le `eps` le moins bruité donnant au moins
deux groupes, puis sur le `eps` médian. DBSCAN tourne dans l'espace réduit par
ACP (95 % de variance) : les densités perdent leur sens en grande dimension.

> **Lecture de DBSCAN.** DBSCAN ne cherche pas `k` groupes, il cherche des zones
> denses séparées par du vide. Sur un nuage qui est un continuum — ce qui est le
> cas du jeu de substitution, un gradient nord-sud régulier — il renvoie une ou
> deux classes et beaucoup de bruit. Ce n'est pas un échec du code : c'est le
> résultat, à rapporter tel quel. Comparer sa silhouette à celle de K-Means a
> peu de sens, les deux ne répondent pas à la même question.

**Métriques** : silhouette, Calinski-Harabasz, Davies-Bouldin, nombre de
clusters, part de bruit.

**Évolution sur la décennie** : effectif, part, température moyenne et anomalie
de chaque cluster année par année (`evolution`), et matrice de transition des
points entre la première et la dernière année (`transitions`).

Sorties : `data/processed/unsupervised.parquet`, `data/processed/clusters_by_year.parquet`.

## Docker

L'image sert JupyterLab sur le port lu dans `PORT` (8888 par défaut) et embarque
le jeu de substitution déjà prétraité :

```bash
docker build -t era5-dm .
docker run -e PORT=8888 -p 127.0.0.1:8888:8888 era5-dm
```

JupyterLab y est démarré **sans jeton** : à n'exposer que sur `127.0.0.1`.
