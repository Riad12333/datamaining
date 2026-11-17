
# Guide de Présentation — Étape 1 : EDA & Prétraitement Fiabilisé
**Projet Data Mining — USTHB (M2 SII)**  
**Dataset : ERA5-Land Algérie (2015 → 2026)**

---

## 🎯 En 30 secondes : Le pitch du projet
> *« Notre objectif dans cette Étape 1 est de préparer rigoureusement 11 années de réanalyses climatiques ERA5-Land sur l'Algérie pour prédire la classe de température du mois suivant ($M+1$). Chaque décision de nettoyage est justifiée par des mesures statistiques réelles, sans aucune fuite de données (*Data Leakage*), et toutes les anomalies du serveur Copernicus ont été diagnostiquées et corrigées. »*

---

## 🧭 Plan de Présentation (4 Actes logiques)

```
[ ACTE 1 : Cadre & Frontière ]   --> 2×2 pixels, masque Algérie, 5 351 points, 0 NaN
                 ↓
[ ACTE 2 : Problème ECMWF 2022-2024] --> Problème connu (variables cumulées), facteur constant par variable
                 ↓
[ ACTE 3 : Sélection & Features] --> Spearman, retrait slhf & ro, log1p(tp), wind_speed & rnet
                 ↓
[ ACTE 4 : Préparation ML ]      --> Cible M+1, 3 classes équi-largeur, StandardScaler sur Train
```

---

## 📊 Matrice des Figures et Tableaux générés

Voici comment relier chaque fichier généré à l'argumentation devant la prof :

| Étape / Question prof | Fichier Tableau (`reports/tables/`) | Fichier Figure (`reports/figures/`) | Message clé à donner à l'oral |
|---|---|---|---|
| **1. Choix du masque spatial (Point a)** | `01_comparaison_methodes_selection.csv` | `01_comparaison_methodes_frontiere.png` | *« La Méthode A laissait des trous ou ignorait la frontière. Avec la Méthode B (`count_dz >= 2`), on garde 5 351 points de grille avec strictement 0 NaN. »* |
| **2. Données erronées 2022–2024 (Point d)** | `01_anomalies.csv` (108 corrections) | `01_series_temporelles_brutes.png`<br>`01_series_temporelles_corrigees.png` | *« Il s'agit d'un problème connu d'ECMWF (variables cumulées, 2022-09 à 2024-02, non corrigé dans le produit standard). Nous appliquons un facteur constant par variable calculé sur le train sain, évitant la méthode circulaire et préservant les anomalies réelles mois par mois. »* |
| **3. Variabilité & Asymétrie (Points b & c)** | `01_variabilite_asymetrie.csv` | `01_asymetrie_avant_apres.png`<br>`01_boxplots_mensuels.png` | *« `ro` a 77.5% de zéros (désert). La pluie `tp` est très asymétrique : le log1p compresse la traîne droite sans supprimer les vrais orages sahariens. »* |
| **4. Corrélations (Point e)** | `01_correlations_spearman.csv`<br>`01_fortes_correlations.csv` | `01_correlations_spearman.png` | *« Calculé avec Spearman (non-linéaire). Élimination de `slhf` (colinéarité 1.000 avec `e`) et de `ro` (0.976 avec `tp`). »* |
| **5. Cible & Classes (Points 2, 4, 6)** | `01_repartition_classes.csv`<br>`01_repartition_classes_par_annee.csv` | `02_bornes_et_classes.csv` | *« Bornes équi-largeur calculées sur le Train seul ([0.91, 14.33, 27.74, 41.16] °C). Train = 642k lignes, Test = 107k lignes. »* |

---

## 🗣️ Fiche Antisèche : Réponses aux 5 Questions Pièges de la Prof

### Q1. « Pourquoi avez-vous éliminé `slhf` et `ro`, mais gardé `sp` ? »
> **Réponse :**  
> *« `slhf` est physiquement identique à l'évaporation `e` ($\rho = 1.000$). `ro` est nul à 77.5% et redondant avec la pluie ($\rho = 0.976$). Les garder créerait une multicolinéarité néfaste. En revanche, `sp` (pression) a un faible coefficient de variation (0.035) mais une grande amplitude physique (788 à 1 037 hPa) qui capture directement le relief de l'Atlas et du Hoggar. »*

### Q2. « Pourquoi avoir utilisé la corrélation de Spearman et non Pearson ? »
> **Réponse :**  
> *« Pearson suppose des relations linéaires et des distributions gaussiennes. Or la pluie `tp`, l'évaporation `e` et le rayonnement sont fortement asymétriques et non-linéaires. Spearman est fondé sur les rangs : il est robuste aux valeurs extrêmes et mesure la monotonie sans biais. »*

### Q3. « Comment avez-vous traité les données erronées de 2022–2024 ? »
> **Réponse :**  
> *« C'est un problème connu d'ECMWF (variables cumulées, 2022-09 à 2024-02, non corrigé dans le produit standard). La cause exacte n'étant pas documentée officiellement, nous appliquons un facteur d'échelle constant par variable (l'inverse de la médiane des ratios calculée sur les années saines du train). Cette méthode de secours évite la circularité d'une correction mois par mois et préserve la dispersion et les anomalies climatiques réelles. »*

### Q4. « Y a-t-il un risque de fuite de données (*Data Leakage*) dans votre prétraitement ? »
> **Réponse :**  
> *« Absolument aucun. Tout ce qui apprend un paramètre (les 3 bornes de discrétisation de la cible, la moyenne et l'écart-type du `StandardScaler`, la matrice de Spearman et le choix des variables à écarter) a été calculé exclusivement sur le jeu d'entraînement (< 2025). Le jeu de test (2025–2026) est resté totalement aveugle. »*

### Q5. « Quels sont les fichiers finaux générés pour la suite ? »
> **Réponse :**  
> *« Dans `data/processed/` : `train.parquet` (642 120 lignes), `test.parquet` (107 020 lignes), `scaler.joblib` et `features.json` contenant les 12 features prêtes pour les algorithmes supervisés de l'Étape 2. »*
