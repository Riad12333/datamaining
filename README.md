# Analyse climatique ERA5 — température et rayonnement thermique net

Projet de data mining sur les réanalyses **ERA5-Land mensuelles** (Copernicus CDS),
variables `t2m` (température à 2 m) et `str` (rayonnement thermique net de surface),
sur l'Algérie et sa marge maritime, 1991–2024.

Le dépôt couvre trois étapes : nettoyage/exploration, apprentissage supervisé,
clustering.

## Structure

```
data/raw/         données brutes (NetCDF du CDS, ou CSV de substitution)
data/processed/   tables nettoyées et agrégées (parquet)
notebooks/        01_eda · 02_supervise · 03_clustering
src/              config · download · preprocessing · supervised · clustering
```

## Installation

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Récupération des données

Avec une clé Copernicus CDS dans `~/.cdsapirc` :

```bash
python -m src.download
```

Sans clé (jeu de substitution de même structure, pour faire tourner le pipeline) :

```bash
python -m src.download --synthetic
```

## Les trois étapes

### 3. Exploration et nettoyage — `src/preprocessing.py`, `notebooks/01_eda.ipynb`

- **Valeurs manquantes** : dans ERA5-Land les points en mer sont masqués. `missing_report`
  classe chaque point de grille (`terre` / `partiel` / `mer`) selon son taux de NaN ;
  `drop_sea_points` supprime les points entièrement manquants et interpole les trous
  isolés le long de la série temporelle de chaque point.
- **Conversion des unités** : `t2m` passe des kelvins aux °C (−273.15) ; `str` passe des
  J m⁻² (cumul journalier des moyennes mensuelles) aux W m⁻² (÷ 86 400). Les conversions
  sont idempotentes — relancer le script ne convertit pas deux fois.
- **Distributions et corrélations** : histogrammes, cycle saisonnier par mois, matrice de
  corrélation entre `t2m`, `str`, latitude, longitude et mois.

Sortie : `data/processed/era5_clean.parquet` (format long : `time`, `latitude`,
`longitude`, `t2m`, `str`, `year`, `month`).

### 4. Préparation supervisée — `src/supervised.py`, `notebooks/02_supervise.ipynb`

- **Cible** : `t2m` décalée d'un mois (`shift(-1)` à l'intérieur de chaque point de grille),
  puis **discrétisée en 3 intervalles de même largeur** (`froid` / `tempere` / `chaud`) via
  `np.linspace` sur l'étendue — équi-largeur, pas équi-effectif.
- **Split chronologique** : les 5 dernières années en test, le reste en train. Aucun
  mélange aléatoire, pour ne pas prédire le passé avec le futur.
- **Point de méthode** : les bornes des intervalles sont calculées **sur le train seul**
  puis appliquées au test, bornes extrêmes ouvertes (`±inf`) pour absorber les valeurs
  hors étendue. Découper sur l'ensemble complet ferait fuir l'information du futur.
- **Variables explicatives** : `t2m`, `str`, retards `t2m_lag1`, `t2m_lag2`, `str_lag1`,
  position (latitude, longitude) et encodage cyclique du mois (sin/cos).
- Modèles de référence : classe majoritaire, arbre de décision, forêt aléatoire.

Sortie : `data/processed/supervised.parquet` (colonne `split` = train/test).

### 5. Préparation non supervisée — `src/clustering.py`, `notebooks/03_clustering.ipynb`

Une ligne = **un couple (année, point de grille)**, soit 35 colonnes :

- `t2m_01` … `t2m_12` et `str_01` … `str_12` : les 12 valeurs mensuelles ;
- par variable, les features de tendance : `_mean` (moyenne), `_std` et `_amplitude`
  (écart-type et max−min, deux lectures de l'« écart »), `_slope` (pente de la régression
  linéaire sur les 12 mois, en unité par mois).

Les couples (année, point) incomplets sont écartés. Le clustering enchaîne
standardisation → ACP (95 % de variance) → K-means, avec courbe du coude et silhouette
pour le choix de `k`.

Sortie : `data/processed/unsupervised.parquet`.

## Lancer tout le pipeline

```bash
python -m src.download --synthetic
python -m src.preprocessing
python -m src.supervised
python -m src.clustering
```

## Docker

L'image sert JupyterLab (sans jeton) sur le port lu dans `PORT`, défaut 8888 :

```bash
docker build -t era5 .
docker run -e PORT=8888 -p 8888:8888 era5
```
