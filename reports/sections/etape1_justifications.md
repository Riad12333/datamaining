# Étape 1 — Justification de l'Ordre des Prétraitements et Décisions Méthodologiques

**Projet :** Data Mining — USTHB (ERA5-Land Algérie 2015 → 2026)  
**Auteur :** Ingénieur Data Science  
**Date :** 16 novembre 2026  

---

## 1. Règle d'or méthodologique : Prévention stricte de toute fuite d'information (*Data Leakage*)

La conception de la chaîne de prétraitement repose sur une frontière méthodologique fondamentale entre :
1. **Les opérations déterministes et physiques :** indépendantes de la distribution globale des données et ne modifiant pas les distributions relatives (découpage géographique vectoriel, conversions d'unités physiques, correction d'un défaut de transmission d'un capteur/producteur, décalage temporel pour construire la cible, transformations mathématiques fixes monotones sans paramètre comme $\log(1+x)$). Ces étapes sont exécutées en amont du découpage Train/Test.
2. **Les opérations statistiques d'apprentissage :** celles qui estiment un paramètre à partir des données (moyenne et écart-type pour le centrage-réduction, quantiles ou bornes minimales/maximales pour la discrétisation, choix de sélection/élimination de features selon les corrélations). **Toutes ces étapes sont strictement ajustées sur le sous-ensemble d'entraînement (*Train set* : années 2015 à 2024)**, puis appliquées sans modification au sous-ensemble de test (*Test set* : années 2025 et 2026).

Aucune information future (test) n'intervient dans le calibrage d'un scaler, d'une frontière de discrétisation ou d'un seuil statistique.

---

## 2. Tableau ordonné des étapes de prétraitement et justifications

| Ordre | Étape | Nature | Pourquoi à cette place précise ? |
| :---: | :--- | :--- | :--- |
| **1** | **Extraction du NetCDF brut (`open_raw`)** | Physique | Accès aux dimensions natives (0.1°, 141 pas de temps, 11 variables météorologiques). Gestion de la coordonnée `expver` (fusion ERA5/ERA5T). |
| **2** | **Masquage géographique à 0.1° puis réduction 2×2 (Méthode B)** | Déterministe / Spatial | Doit être fait **avant** toute moyenne spatiale. Masquer à la résolution native (0.1°) évite de contaminer les pixels côtiers algériens avec de l'eau de mer ou des territoires limitrophes. Le filtre `min_pixels >= 2` conserve les mailles majoritairement algériennes ($\ge 50\%$) et garantit l'absence totale de NaN. |
| **3** | **Conversion des unités physiques (`convert_units`)** | Déterministe / Physique | Conversion immédiate ($K \to °C$, $J/m^2 \to W/m^2$, $m \to mm/jour$, $Pa \to hPa$) pour manipuler des grandeurs physiques compréhensibles lors de toute l'analyse exploratoire et des règles d'écrêtage. |
| **4** | **Correction du problème d'échelle ECMWF 2022–2024 (`correct_ecmwf_anomalies`)** | Physique / Métrologique | Doit être corrigé avant l'analyse des distributions, corrélations et calculs de cibles. Corriger l'échelle des variables cumulées par un facteur constant par variable (médiane des 18 ratios calculée sur le train sain) rétablit le niveau physique réel tout en préservant les anomalies climatiques mois par mois, sans circularité. |
| **5** | **Construction de la cible supervisée à $M+1$ (`build_frame`)** | Déterministe / Temporel | Le décalage temporel $t+1$ par point de grille (`shift(-1)`) requiert une série temporelle complète et continue. Toute suppression de ligne à ce stade romprait la consécutivité des mois et invaliderait la cible du mois précédent. |
| **6** | **Feature extraction physique (`wind_speed`, `rnet`)** | Déterministe / Physique | Dérivation de grandeurs énergétiques et dynamiques explicables ($\text{rnet} = \text{ssr} + \text{str}$, $\text{wind\_speed} = \sqrt{u_{10}^2 + v_{10}^2}$) directement à partir des variables physiques corrigées. |
| **7** | **Transformations monotones sans paramètre (`log1p`)** | Déterministe / Distributionnel | Résolution de la forte asymétrie droite de la pluie (`tp`) et de l'évaporation (`-e`). Comme $\log(1+x)$ ne dépend d'aucun paramètre appris, elle ne crée aucune fuite et stabilise les échelles avant modélisation. |
| **8** | **Split chronologique Train / Test (`chronological_split`)** | Méthodologique | Césure temporelle stricte : Train (2015–2024, 10 ans) vs Test (2025–2026, 2 ans). C'est la frontière étanche de fuite d'information. |
| **9** | **Sélection de features & suppression colinéarité (`DROPPED_FEATURES`)** | Statistique (calé sur Train) | La matrice des corrélations de Spearman est calculée sur le **Train seul**. On y constate $\rho(slhf, e) = 1.000$ et $\rho(ro, tp) = 0.976$. Le choix d'écarter `slhf` et `ro` est validé sur le Train. |
| **10** | **Discrétisation équi-largeur de la cible (`discretize`)** | Statistique (calé sur Train) | Les bornes des 3 intervalles de même largeur $[0.91, 14.33, 27.74, 41.16]\,°C$ sont calculées à partir des extrêmes du **Train seul**. Les bornes extrêmes sont étendues à $\pm\infty$ pour absorber les dépassements du Test. |
| **11** | **Standardisation des features (`StandardScaler`)** | Statistique (calé sur Train) | Moyennes et écarts-types sont estimés **uniquement sur le Train**, puis appliqués au Test pour les modèles sensibles aux échelles (SVM, KNN). |

---

## 3. Registre exhaustif des décisions prises sur les données

| Variable / Étape | Problème constaté | Chiffre mesuré | Décision prise | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Frontière Nord (Méthode A vs B)** | Les mailles côtières 2×2 contiennent de la mer ; la méthode A (centroïde) perd des mailles valides ou inclut des fractions marines. | Méthode A : 5 316 points.<br>Méthode B : 5 351 points (`min_pixels=2`), 0 NaN.<br>Méthode B (`min_pixels=1`) : 2 points côtiers 100% NaN (282 NaNs). | **Adoption de la Méthode B avec `min_pixels=2`**. | Réduction 2×2 appliquée sur les seuls pixels terrestres algériens à 0.1°. Règle de majorité ($\ge 50\%$) : gagne 35 points frontaliers valides sans générer le moindre NaN. |
| **Anomalie 2022–2024** | Chute brutale de ~50% du rayonnement (`ssr`, `str`) et des flux cumulés entre sept. 2022 et fév. 2024. | Ratio médian `str` = 0.4903 ($F_{\text{const}} = 2.0395$).<br>Ratio médian `ssr` = 0.5349 ($F_{\text{const}} = 1.8696$).<br>Z-score spatial < -3.1 sur 6 mois consécutifs en 2023. | **Facteur CONSTANT par variable** : multiplication par $F_{\text{const}}(v) = 1 / \text{médiane}(r(v))$ calculé sur les années saines du train. | Problème connu d'ECMWF (variables cumulées, 2022-09 à 2024-02, non corrigé dans le produit standard). La cause exacte n'est pas documentée. L'application d'un facteur constant évite la méthode circulaire ($moyenne = normale$) et préserve la dispersion mois par mois (std post-correction : 0.026 pour `ssr`, 0.034 pour `str`, 1.022 pour `tp`). *Limite :* suppose une erreur relative homogène sur la période. |
| **Variables d'état (`t2m`, `d2m`, `sp`, `u10`, `v10`)** | Vérification de potentielles anomalies 2022–2024. | Ratios mensuels $\in [0.992, 1.011]$, Z-scores $\in [-0.5, +0.5]$. | **Conservation intégrale sans modification**. | Les variables d'état instantanées n'étaient pas soumises au bug d'accumulation de l'ECMWF. |
| **`slhf` (Flux de chaleur latente)** | Colinéarité parfaite avec l'évaporation totale `e`. | Corrélation de Spearman $\rho = 1.0000$ sur le Train. | **Écartée (`DROPPED_FEATURES`)**. | Identité physique stricte : $\text{SLHF} = L_v \cdot \rho_w \cdot E$. Conserver `e` (en mm/jour) est plus intuitif pour l'interprétation métier. |
| **`ro` (Ruissellement)** | Quasi-constance et forte colinéarité avec la pluie `tp`. | 77.5% de zéros stricts, IQR/étendue = 0.00007 (< 0.01), $\rho(\text{ro}, \text{tp}) = 0.9763$. | **Écartée (`DROPPED_FEATURES`)**. | En milieu aride saharien, le ruissellement est nul 99% du temps et redondant avec les précipitations. |
| **`tp` (Précipitations)** | Distribution extrêmement asymétrique à queue lourde à droite. | Skewness initiale = +6.40, 15.0% de pseudo-outliers selon la règle IQR standard. | **Transformation $\log(1 + \text{tp})$ appliquée**. | Réduit l'asymétrie à +3.46. Les fortes pluies sont des événements désertiques réels, indispensables à conserver. |
| **`e` (Évaporation)** | Variable négative orientée vers le haut, très asymétrique. | Skewness initiale = -4.33, 16.5% de pseudo-outliers IQR. | **Transformation $\log(1 - \text{e})$ appliquée**. | Réduit l'asymétrie à +2.99 tout en conservant la monotonicité stricte. |
| **`sp` (Pression de surface)** | Faible coefficient de variation ($CV = 0.035$). | Étendue : 788 à 1 037 hPa (écart-type 33.5 hPa). | **Conservée sans transformation**. | La pression de surface capture le gradient altimétrique et topographique (du littoral au mont Tahat à 3 000 m), déterminant pour la température. |
| **Pseudo-outliers (règle IQR vs Z robuste)** | La règle standard $1.5 \times \text{IQR}$ marque à tort ~15% des points comme outliers dans le Sahara. | Z robuste $> 5$ (médiane/MAD par point et mois) ne touche que 0.48% de `t2m` et 0.85% de `ssr`. | **Aucune suppression de ligne**. Valeurs aberrantes ponctuelles interpolées dans le temps. | L'IQR suppose la normalité et échoue sur des distributions climatiques bimodales ou à queues épaisses. |
| **Feature Extraction : `wind_speed`** | Les composantes $u_{10}$ et $v_{10}$ représentent la direction du vent, mais la force brute régit les échanges thermiques. | Moyenne = 1.90 m/s, distribution quasi-normale. | **Ajout de $\sqrt{u_{10}^2 + v_{10}^2}$**. | Grandeur physique scalaire invariante par rotation, décisive pour le refroidissement convectif. |
| **Feature Extraction : `rnet`** | `ssr` et `str` sont corrélées négativement ($\rho = -0.81$). | Moyenne = 47.92 W/m². | **Ajout de $\text{ssr} + \text{str}$**. | Représente le bilan radiatif net à la surface, moteur premier du bilan d'énergie du sol. |
| **Discrétisation de la cible** | Consigne d'intervalles de même largeur (équi-largeur). | Bornes Train : $[0.91, 14.33, 27.74, 41.16]\,°C$ (largeur = 13.41 °C). | **3 classes : froid, tempéré, chaud**. | Calculé sur le Train seul. Le déséquilibre des effectifs (17.8% froid, 43.3% tempéré, 38.9% chaud) est la conséquence naturelle de l'équi-largeur en climat méditerranéen/saharien. |

---

## 4. Limites de l'étude et perspectives

1. **Split temporel unique (absence de validation croisée temporelle) :**  
   L'évaluation repose sur une coupure chronologique fixe (2015–2024 en entraînement, 2025–2026 en test). Bien qu'elle simule parfaitement le déploiement opérationnel, les années 2025 et 2026 ont été marquées par des anomalies de chaleur historiques en Méditerranée occidentale, ce qui explique une part de classe « chaud » plus élevée dans le jeu de test (45.7% contre 38.9% dans le train). Un schéma de *TimeSeriesSplit* ou *Rolling-Origin* permettrait d'estimer la variance temporelle des modèles.
2. **Année 2026 incomplète :**  
   Le jeu NetCDF fourni s'arrête en septembre 2026 (9 mois disponibles). Pour les calculs de tendance annuelle et le clustering annuel (étape 3), ces mois sont traités avec précaution (`complete_years`) pour ne pas biaiser les moyennes annuelles par l'absence du dernier trimestre automnal/hivernal.
3. **Absence de la variable `sshf` (*Surface Sensible Heat Flux*) :**  
   La variable de flux de chaleur sensible n'était pas présente dans l'export NetCDF brut téléchargé. Conformément à la consigne, aucun jeu de substitution artificiel n'a été créé, et la chaîne opère rigoureusement avec les 10 prédicteurs réels disponibles complétés par le bilan radiatif net et la vitesse du vent.


## 5. Références bibliographiques

1. **Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).** *A density-based algorithm for discovering clusters in large spatial databases with noise.* In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), pp. 226–231. AAAI Press.
2. **Fayyad, U., Piatetsky-Shapiro, G., & Smyth, P. (1996).** *From Data Mining to Knowledge Discovery in Databases.* AI Magazine, 17(3), 37–54.
3. **Muñoz-Sabater, J., et al. (2021).** *ERA5-Land: A state-of-the-art global reanalysis dataset for land applications.* Earth System Science Data, 13(9), 4349–4383.
