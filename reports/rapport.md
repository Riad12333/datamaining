# Prédiction de la classe de température à 2 m sur l'Algérie

**Projet de Data Mining** — USTHB / FI / IASD — M2 SII, 2026-2027
Encadrement : Dr. Houacine & Houacine

Auteur : ‹NOM PRÉNOM› — Date : ‹DATE›
Dépôt : https://github.com/raoufyd/dm

> **⚠ À SUPPRIMER AVANT REMISE.** Ce document est un squelette. Chaque marque
> `‹…›` doit être remplacée par un résultat obtenu sur les **vraies** données
> ERA5-Land. Les tableaux et figures sont produits par
> `python -m scripts.export_assets` dans `reports/tables/` et `reports/figures/`.
> Vérifier `reports/PROVENANCE.txt` : tant qu'il indique « jeu de substitution »,
> aucun chiffre de ce rapport n'est publiable.

---

## Résumé

‹200 mots : objectif, données, méthode, meilleur modèle et son score, principal
enseignement du clustering sur la décennie.›

---

## 1. Introduction

### 1.1 Contexte

La crise climatique impose des prévisions fiables pour orienter les politiques
d'adaptation. Les modèles physiques classiques sont peu performants à basse
résolution spatiale et deviennent coûteux à haute résolution. Les méthodes
d'apprentissage automatique offrent une alternative à la fois efficace et peu
coûteuse en calcul.

### 1.2 Objectifs

1. collecter et prétraiter des données climatiques ;
2. prédire par apprentissage supervisé la **classe** de température à 2 m, la
   variable continue étant d'abord découpée en trois intervalles de même largeur ;
3. identifier par clustering des regroupements naturels dans les données ;
4. évaluer les modèles avec des métriques standard.

### 1.3 Zone d'étude et données

| | |
|---|---|
| Zone | Algérie, emprise `‹AREA de src/config.py›` |
| Période | 2015 → ‹dernière année complète› |
| Source | ERA5-Land monthly averaged reanalysis (Copernicus CDS) |
| Résolution native | 0,1° |
| Résolution après agrégation | 0,4° (blocs 4×4) |
| Variables | 12 (11 explicatives + la cible `t2m`) |

---

## 2. Étape 1 — Analyse et prétraitement

### 2.1 Réduction de la résolution spatiale

L'énoncé impose de diviser la résolution par quatre. L'agrégation est faite par
moyenne de blocs 4×4 (`xarray.Dataset.coarsen`), ce qui fait passer la grille de
0,1° à 0,4°, soit de ‹N› à ‹M› points de grille.

Deux choix techniques méritent d'être justifiés :

- `boundary="trim"` écarte les lignes et colonnes surnuméraires plutôt que de
  former un bloc partiel : un bloc de 2 points n'aurait pas la même variance
  d'échantillonnage qu'un bloc de 16 et introduirait un biais en bordure ;
- la moyenne ignore les valeurs manquantes, de sorte qu'un bloc à cheval sur la
  côte conserve la moyenne de ses points terrestres au lieu d'être entièrement
  annulé par un unique point marin.

**Compromis.** L'agrégation divise par 16 le volume de données et lisse le bruit
local, au prix d'une perte de détail orographique : à 0,4°, le contraste entre
l'Atlas tellien et les hauts plateaux est fortement atténué.

### 2.2 Valeurs manquantes

ERA5-Land n'est défini que sur les terres émergées : les points marins sont
manquants sur toute la période. Deux régimes sont donc distingués.

| régime | critère | traitement |
|---|---|---|
| hors domaine | > 99 % de NaN sur `t2m` | point supprimé |
| trou isolé | série par ailleurs complète | interpolation linéaire temporelle, au sein du point |

Après nettoyage : ‹N› points de grille retenus, ‹M› lignes.

**Figure 1** — `reports/figures/01_masque_terre_mer.png` : points conservés et
points écartés.

### 2.3 Conversion des unités

| variables | de | vers | opération |
|---|---|---|---|
| `t2m`, `d2m` | K | °C | − 273,15 |
| `slhf`, `ssr`, `str`, `sshf` | J m⁻² (cumul journalier) | W m⁻² | ÷ 86 400 |
| `ro`, `e`, `tp` | m | mm/jour | × 1000 |
| `sp` | Pa | hPa | ÷ 100 |

Ces conversions ne changent rien aux performances des modèles ; elles servent
l'interprétabilité. Elles sont implémentées de façon idempotente (test préalable
sur la médiane), de sorte qu'une seconde exécution du script ne convertit pas
deux fois.

### 2.4 Analyse exploratoire

**Tableau 1** — `reports/tables/01_statistiques_descriptives.md`.

**Figure 2** — `reports/figures/01_distributions.png` : distribution des 12
variables. ‹Commenter : asymétries, variables bornées comme `ro` et `tp`,
bimodalité éventuelle liée au contraste Tell / Sahara.›

**Figure 3** — `reports/figures/01_correlations.png`. ‹Commenter les corrélations
notables : `d2m`/`t2m`, signe opposé de `ssr` et `str`, rôle de la saisonnalité
comme facteur commun. Discuter la colinéarité et ses conséquences pour les
modèles linéaires — sans effet pour les arbres.›

**Figure 4** — `reports/figures/01_saisonnalite_tendance.png` : cycle saisonnier
et moyenne annuelle. Tendance mesurée : ‹X› °C/an.

---

## 3. Étape 2 — Apprentissage supervisé

### 3.1 Construction des instances

Chaque instance décrit un couple (point de grille, mois) :

- les **11 variables explicatives** imposées : `d2m`, `slhf`, `ssr`, `str`,
  `sshf`, `ro`, `e`, `u10`, `v10`, `sp`, `tp` ;
- le **mois** et l'**année** ; le mois est en outre encodé de façon cyclique
  (`sin`, `cos`), car décembre et janvier sont adjacents dans le temps mais
  distants de 11 en valeur numérique.

La **cible** est la température à 2 m du **mois suivant**, obtenue par
`shift(-1)` appliqué *à l'intérieur de chaque point de grille* — un décalage
global ferait déborder la dernière valeur d'un point sur le point suivant. Un
contrôle supplémentaire invalide la cible lorsque les deux mois ne se suivent
pas réellement, cas possible après le traitement des valeurs manquantes.

### 3.2 Discrétisation en trois intervalles de même largeur

L'étendue de la cible est divisée en trois intervalles de **largeur égale**
(`np.linspace`), et non en trois quantiles. La conséquence directe est un
**déséquilibre des classes**, qui justifie de reporter la F1 macro et l'accuracy
équilibrée en plus de l'accuracy.

**Tableau 2** — `reports/tables/02_bornes_et_classes.md` : bornes en °C et part
de chaque classe dans le train et le test.

### 3.3 Partitionnement chronologique et prévention des fuites

Les ‹TEST_YEARS› dernières années constituent le test, le reste le train. Aucun
mélange aléatoire : un tirage au hasard placerait janvier 2024 en apprentissage
et février 2024 en test, et le modèle bénéficierait d'une information quasi
contemporaine de ce qu'il doit prédire.

> **Choix de méthode.** Les bornes de discrétisation sont calculées sur le
> **train seul**, puis appliquées au test. Les calculer sur l'ensemble complet
> introduirait dans l'apprentissage une information sur l'étendue des
> températures futures — une fuite de données. Les bornes extrêmes sont donc
> ouvertes (−∞, +∞) pour absorber les valeurs de test hors étendue.

De même, le `StandardScaler` est encapsulé dans un `Pipeline` scikit-learn :
il est ajusté sur le train à chaque `fit`, jamais sur le test.

### 3.4 Modèles

| modèle | rôle | paramètres principaux |
|---|---|---|
| classe majoritaire | référence | — |
| KNN | ‹…› | k = 15, pondération par distance, données standardisées |
| SVM | ‹…› | noyau RBF, C = 10, données standardisées |
| Forêt aléatoire | ‹…› | 300 arbres, `min_samples_leaf` = 2 |
| AdaBoost | ‹…› | 200 souches de profondeur 3, `learning_rate` = 0,5 |
| Vote dur | ‹…› | KNN + SVM + forêt, majorité |
| Vote souple | ‹…› | KNN + SVM + forêt, moyenne des probabilités |

KNN et SVM raisonnent sur des distances et exigent la standardisation : `sp` en
hPa (ordre 10³) écraserait `u10` en m s⁻¹ (ordre 1). Les arbres, qui comparent
seuil par seuil et variable par variable, en sont dispensés.

**Vote dur contre vote souple.** Le vote dur compte une voix par modèle ; le vote
souple moyenne les probabilités et pondère donc implicitement par la confiance.
Le second exige que chaque modèle expose des probabilités, ce qui impose
`SVC(probability=True)` et un calibrage de Platt interne nettement plus coûteux.

> **Sous-échantillonnage, et pourquoi il est déclaré.** La complexité de `SVC`
> est entre O(n²) et O(n³) ; l'entraînement sur ‹N› lignes est hors de portée.
> Tous les modèles sont donc entraînés sur le **même** sous-échantillon
> aléatoire de ‹SVM_MAX_TRAIN› lignes, afin que la comparaison reste à effectif
> égal — comparer un SVM bridé à une forêt entraînée sur la totalité fausserait
> le classement. La taille effective est journalisée à chaque exécution.

### 3.5 Résultats

**Tableau 3** — `reports/tables/02_comparaison_modeles.md` : accuracy, accuracy
équilibrée, F1 macro et pondérée, kappa de Cohen, temps d'ajustement.

**Figure 5** — `reports/figures/02_comparaison_modeles.png`.

Meilleur modèle : ‹NOM› (F1 macro ‹X›, accuracy ‹Y›), contre ‹Z› pour la
référence. ‹L'écart à la référence est le chiffre à commenter : une accuracy
élevée dans l'absolu s'explique en grande partie par le cycle saisonnier, qui
rend la classe du mois suivant largement prévisible.›

**Figure 6** — `reports/figures/02_matrice_confusion.png`. ‹Commenter : les
confusions se concentrent-elles entre classes adjacentes (froid ↔ tempéré,
tempéré ↔ chaud) ? Une confusion froid ↔ chaud serait anormale.›

**Figure 7** — `reports/figures/02_importance_variables.png`. ‹Commenter les
variables dominantes et leur plausibilité physique.›

### 3.6 Discussion

‹Limites : la colinéarité entre variables ; la dépendance spatiale entre points
voisins, qui rend les instances non indépendantes et peut optimiser le score ;
l'absence de validation croisée temporelle ; le sous-échantillonnage.›

---

## 4. Étape 3 — Clustering

### 4.1 Construction des instances

Une instance est un couple **(année, point de grille)**. Les couples comptant
moins de 12 mois sont écartés. Total : ‹N› instances, ‹M› descripteurs.

| groupe | descripteurs |
|---|---|
| valeurs mensuelles | `t2m_01`…`t2m_12`, `str_01`…`str_12` |
| tendance intra-annuelle | `_mean`, `_std`, `_amplitude`, `_pente` (pente OLS sur les 12 mois) |
| tendance long terme | `_anomalie`, `_derive` |

Les deux derniers descripteurs répondent à la demande de l'énoncé de *capter les
tendances climatiques de long terme* :

- `_anomalie` : écart de la moyenne annuelle du point à sa **normale**, estimée
  sur les ‹BASELINE_YEARS› premières années ;
- `_derive` : pente des moyennes annuelles de ce point sur toute la période.

Sans ces deux descripteurs, deux années de même *forme* saisonnière seraient
indiscernables même si l'une est globalement plus chaude d'un degré — or c'est
précisément le signal recherché.

### 4.2 Deux précautions indispensables à l'analyse d'évolution

**Standardisation globale.** Les variables sont centrées-réduites sur
l'ensemble des années, non année par année. Une standardisation annuelle
effacerait le décalage inter-annuel, c'est-à-dire exactement la quantité que
l'analyse d'évolution cherche à mesurer.

**Renumérotation canonique des étiquettes.** Les étiquettes de K-Means sont
arbitraires : le cluster 0 de 2015 n'a aucun lien avec le cluster 0 de 2016. Les
clusters sont donc renumérotés par température moyenne croissante, ce qui rend
comparables les cartes successives et donne un sens à la matrice de transition.

### 4.3 Choix de k

**Figure 8** — `reports/figures/03_choix_k.png` : courbe du coude et silhouette.
Valeur retenue : k = ‹K›, justification : ‹…›.

### 4.4 Algorithmes

- **K-Means** — partitionnement par minimisation de l'inertie intra-classe.
- **AGNES** — agglomératif ascendant, lien de Ward.
- **DBSCAN** — densité. Le paramètre `eps` est choisi sur la courbe des
  k-distances : balayage des centiles, sélection du `eps` maximisant la
  silhouette sous contrainte d'une part de bruit ≤ ‹DBSCAN_MAX_NOISE›, avec
  repli sur le `eps` le moins bruité donnant au moins deux groupes. DBSCAN opère
  dans l'espace réduit par ACP (95 % de variance) : la notion de densité perd
  son sens en grande dimension.
- **AutoEncoder + K-Means** — un perceptron multicouche de forme (32, ‹b›, 32)
  est entraîné à reconstruire son entrée ; K-Means est ensuite appliqué aux
  activations de la couche goulot. À la différence de l'ACP, qui ne capte que des
  combinaisons linéaires, l'auto-encodeur peut représenter des relations non
  linéaires.

Chaque algorithme est appliqué **indépendamment sur chaque année**.

### 4.5 Résultats

**Tableau 4** — `reports/tables/03_comparaison_algorithmes.md` : nombre de
clusters, part de bruit, silhouette, Calinski-Harabasz, Davies-Bouldin, moyennés
sur les années.

**Figure 9** — `reports/figures/03_metriques_par_annee.png` : stabilité des
métriques dans le temps.

**Figure 10** — `reports/figures/03_cartes_clusters.png` : les quatre
partitionnements pour la dernière année.

**Figure 11** — `reports/figures/03_profils_clusters.png` : profil annuel moyen
de chaque cluster.

> **Comment lire le résultat de DBSCAN.** DBSCAN ne cherche pas k groupes mais
> des zones denses séparées par du vide. Appliqué à un nuage qui est un
> continuum — un gradient régulier sans frontière franche — il renvoie une ou
> deux classes et une forte proportion de bruit. Ce n'est pas un défaut
> d'implémentation : c'est l'information que la structure spatiale est graduelle
> plutôt que séparée en blocs. Comparer sa silhouette à celle de K-Means a donc
> peu de sens, les deux algorithmes ne répondant pas à la même question.
>
> ‹Indiquer ici ce qui est effectivement observé sur les données ERA5 réelles :
> la rupture Tell / Sahara pourrait lui donner prise, contrairement au jeu de
> substitution.›

### 4.6 Évolution sur la décennie

**Tableau 5** — `reports/tables/03_evolution_clusters.md` : effectif, part,
température moyenne et anomalie de chaque cluster, année par année.

**Figure 12** — `reports/figures/03_evolution_clusters.png`.

**Figure 13** — `reports/figures/03_transitions.png` : matrice de transition des
points entre la première et la dernière année.

‹Interprétation attendue : une migration des points vers les clusters plus
chauds et une anomalie moyenne croissante traduiraient un réchauffement.
Quantifier — part des points ayant changé de cluster, sens des transitions,
écart d'anomalie entre la première et la dernière année. Confronter à la
tendance globale mesurée en 2.4 : les deux doivent être cohérentes.›

---

## 5. Conclusion

‹Rappel de la démarche, du meilleur modèle supervisé et de son écart à la
référence, du principal enseignement du clustering. Limites : résolution
dégradée, dépendance spatiale entre instances, horizon de prévision d'un seul
mois, absence de validation croisée temporelle. Pistes : horizons plus longs,
validation croisée glissante, modèles séquentiels.›

---

## 6. Reproductibilité

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m src.download          # nécessite ~/.cdsapirc
python -m src.preprocessing
python -m src.supervised
python -m src.clustering
python -m scripts.export_assets # régénère figures et tableaux
```

Graine aléatoire fixée à `RANDOM_STATE = 42` dans `src/config.py`. Versions des
bibliothèques : `requirements.txt`.

---

## 7. Références

1. Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A. *et al.* (2021). ERA5-Land:
   a state-of-the-art global reanalysis dataset for land applications.
   *Earth System Science Data*, 13(9), 4349–4383.
2. Hersbach, H., Bell, B., Berrisford, P. *et al.* (2020). The ERA5 global
   reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730),
   1999–2049.
3. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A density-based
   algorithm for discovering clusters in large spatial databases with noise.
   *Proceedings of KDD-96*, 226–231.
4. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function.
   *Journal of the American Statistical Association*, 58(301), 236–244.
5. Cortes, C., Vapnik, V. (1995). Support-vector networks. *Machine Learning*,
   20(3), 273–297.
6. Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization of
   on-line learning and an application to boosting. *Journal of Computer and
   System Sciences*, 55(1), 119–139.
7. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
8. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation
   and validation of cluster analysis. *Journal of Computational and Applied
   Mathematics*, 20, 53–65.
9. Xie, J., Girshick, R., Farhadi, A. (2016). Unsupervised deep embedding for
   clustering analysis. *Proceedings of ICML*, 478–487.
10. Pedregosa, F., Varoquaux, G., Gramfort, A. *et al.* (2011). Scikit-learn:
    machine learning in Python. *Journal of Machine Learning Research*, 12,
    2825–2830.
11. Hoyer, S., Hamman, J. (2017). xarray: N-D labeled arrays and datasets in
    Python. *Journal of Open Research Software*, 5(1), 10.

> **À vérifier avant remise.** Chaque référence doit avoir été réellement
> consultée et effectivement citée dans le corps du texte. Retirer celles qui ne
> le sont pas plutôt que de gonfler la liste.
