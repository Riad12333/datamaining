## Étape 1 — Analyse et prétraitement

> Chiffres calculés sur ERA5-Land, régénérés par `python -m scripts.write_section1`.

### 1.1 Jeu de données

Les données proviennent de la réanalyse **ERA5-Land monthly averaged** (Copernicus Climate Data Store), sur l'emprise `[37.5, -9.0, 18.5, 12.5]` (Nord, Ouest, Sud, Est) couvrant l'Algérie et une marge maritime. La période retenue va de **2015 à 2026**, soit **141 pas de temps mensuels**. Douze variables sont extraites : les onze variables explicatives imposées par l'énoncé et la cible `t2m`.

La marge maritime est volontaire : elle produit les valeurs manquantes que la section 1.3 traite.

### 1.2 Réduction de la résolution spatiale

L'énoncé impose de réduire la résolution spatiale d'un facteur quatre en agrégeant les points voisins, soit **4 pixels regroupés** par maille. L'agrégation est réalisée par moyenne de blocs 2×2 (`xarray.Dataset.coarsen`), qui fait passer la grille de **0.10° à 0.20°**, soit de **191×216 = 41 256 points** à **95×108 = 10 260 points** (facteur 4.0).

Deux choix techniques méritent justification.

`boundary="trim"` écarte les lignes et colonnes surnuméraires plutôt que de former un bloc partiel en bordure : un bloc incomplet n'a pas la même variance d'échantillonnage qu'un bloc de 4 points, et le conserver introduirait un biais sur le pourtour du domaine.

La moyenne ignore les valeurs manquantes. Un bloc à cheval sur la côte conserve donc la moyenne de ses seuls points terrestres au lieu d'être annulé par un unique point marin ; un bloc entièrement marin reste manquant et sera écarté à l'étape suivante.

**Compromis assumé.** L'agrégation divise le nombre de points par 4 et lisse le bruit local, au prix d'une perte de détail orographique : à 0.2°, le contraste entre l'Atlas tellien et les hauts plateaux est fortement atténué.

### 1.3 Traitement des valeurs manquantes

ERA5-Land n'est défini que sur les terres émergées : les points marins sont manquants sur toute la période. Sur la grille brute, 8.4% des valeurs de `t2m` sont manquantes. Deux régimes sont distingués, car ils n'appellent pas le même traitement.

| régime | critère | traitement | justification |
|---|---|---|---|
| hors domaine | > 99 % de NaN sur `t2m` | point supprimé | aucune information à récupérer : le point n'est jamais observé |
| trou isolé | série par ailleurs complète | interpolation linéaire temporelle, **au sein du point** | la série mensuelle d'un point est lisse ; interpoler entre deux mois voisins est plus fidèle que supprimer l'instance |

Après nettoyage, **5 490 points de grille** sont retenus, pour **774 090 lignes** (5 490 × 141).

**Figure 1** — `reports/figures/01_masque_terre_mer.png` : répartition spatiale des points conservés et écartés.

### 1.4 Conversion des unités

Les conversions servent l'interprétabilité, non la performance : les modèles utilisés sont insensibles à un changement d'échelle affine. Elles sont implémentées de façon **idempotente** (test préalable sur la médiane), de sorte qu'une seconde exécution du script ne convertisse pas deux fois.

| variables | de | vers | opération |
|---|---|---|---|
| `t2m`, `d2m` | K | °C | − 273,15 |
| `slhf`, `ssr`, `str`, `sshf` | J m⁻² (cumul journalier) | W m⁻² | ÷ 86 400 |
| `ro`, `e`, `tp` | m | mm/jour | × 1000 |
| `sp` | Pa | hPa | ÷ 100 |

Le facteur 86 400 appliqué aux flux mérite un mot : ERA5-Land fournit pour les moyennes mensuelles un **cumul journalier moyen** en J m⁻², que l'on ramène à une puissance moyenne en W m⁻² en divisant par le nombre de secondes dans une journée.

### 1.5 Statistiques descriptives

| variable                          | unité   |   moyenne |   écart-type |     min |   médiane |     max |
|:----------------------------------|:--------|----------:|-------------:|--------:|----------:|--------:|
| `d2m` — point de rosée à 2 m      | °C      |      1.56 |         4.85 |  -17.61 |      1.34 |   23.13 |
| `slhf` — flux de chaleur latente  | W m⁻²   |     -5.86 |        11.75 | -153.2  |     -2.11 |   -0.05 |
| `ssr` — rayonnement solaire net   | W m⁻²   |    156.41 |        49.13 |   31.91 |    161.71 |  312.03 |
| `str` — rayonnement thermique net | W m⁻²   |   -110.04 |        30.16 | -172.64 |   -114.77 |  -24.87 |
| `ro` — ruissellement              | mm/j    |      0    |         0.04 |    0    |      0    |    5.59 |
| `e` — évaporation totale          | mm/j    |     -0.2  |         0.41 |   -5.29 |     -0.07 |   -0    |
| `u10` — vent zonal à 10 m         | m s⁻¹   |     -1.02 |         1.59 |   -6.71 |     -0.99 |    7.47 |
| `v10` — vent méridien à 10 m      | m s⁻¹   |     -0.51 |         1.05 |   -4.81 |     -0.46 |    3.34 |
| `sp` — pression de surface        | hPa     |    951.8  |        33.51 |  788.31 |    958.95 | 1036.72 |
| `tp` — précipitations totales     | mm/j    |      0.17 |         0.48 |    0    |      0.02 |   10.8  |
| `t2m` — température à 2 m         | °C      |     23.88 |         8.75 |    0.91 |     24.21 |   41.16 |

**Figure 2** — `reports/figures/01_distributions.png` : histogrammes des douze variables après conversion.

### 1.6 Corrélations

**Figure 3** — `reports/figures/01_correlations.png` : matrice de corrélation de Pearson.

Les associations les plus fortes sont les suivantes.

| variable A                        | variable B                        |      r |
|:----------------------------------|:----------------------------------|-------:|
| `slhf` (flux de chaleur latente)  | `e` (évaporation totale)          |  1     |
| `ssr` (rayonnement solaire net)   | `str` (rayonnement thermique net) | -0.857 |
| `slhf` (flux de chaleur latente)  | `tp` (précipitations totales)     | -0.765 |
| `e` (évaporation totale)          | `tp` (précipitations totales)     | -0.765 |
| `ssr` (rayonnement solaire net)   | `t2m` (température à 2 m)         |  0.607 |
| `str` (rayonnement thermique net) | `t2m` (température à 2 m)         | -0.592 |

Vis-à-vis de la cible `t2m`, les trois variables les plus informatives sont `ssr` (r = +0.61), `str` (r = -0.59), `d2m` (r = +0.34).

**Conséquence pour la modélisation.** Ces corrélations élevées signalent une forte **colinéarité**. Elle pénaliserait un modèle linéaire, dont les coefficients deviendraient instables et ininterprétables, mais reste sans effet sur les arbres et les forêts, qui sélectionnent une variable à la fois. Elle explique aussi pourquoi l'importance des variables d'une forêt aléatoire doit être lue avec prudence : deux variables redondantes se partagent l'importance au lieu de la cumuler.

### 1.7 Saisonnalité et tendance

**Figure 4** — `reports/figures/01_saisonnalite_tendance.png` : cycle saisonnier mensuel et moyenne annuelle.

Le cycle saisonnier domine la variabilité : la température moyenne passe de **12.0 °C** en janvier à **34.5 °C** en juillet, soit une amplitude de **22.5 °C**.

Les années [2026] sont incomplètes dans le jeu téléchargé et sont **écartées du calcul de tendance** : une année tronquée aux premiers mois a une moyenne biaisée vers le haut, ce qui gonflerait artificiellement la pente.

La régression linéaire des moyennes annuelles sur 11 années donne une tendance de **+0.1070 ± 0.0321 °C/an**, soit **+1.07 °C par décennie**.

Cette tendance est à manier avec précaution : 11 années constituent une série courte au regard de la variabilité interannuelle, et l'incertitude ci-dessus ne tient pas compte de l'autocorrélation temporelle, qui la sous-estime.

### 1.8 Sorties produites

| fichier | contenu |
|---|---|
| `data/processed/era5_clean.parquet` | table nettoyée : 774 090 lignes × 17 colonnes |
| `reports/figures/01_*.png` | les quatre figures ci-dessus |
| `reports/tables/01_*.csv\|.md` | statistiques descriptives et matrice de corrélation |

Reproduction : `python -m src.download && python -m src.preprocessing && python -m scripts.export_assets`.
