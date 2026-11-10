# Présentation — Étape 1 : Analyse & Prétraitement
**TP Data Mining — USTHB (M2 SII)**  
**Guide d'explication du code et des graphiques du Notebook**

---

## 🎯 En une seule phrase : c'est quoi le projet ?
> *« On prend 10 ans de données météo sur l'Algérie (2015–2025) pour prédire par Machine Learning la température du mois prochain et observer comment le climat se réchauffe dans le pays. »*

---

## 🧭 Ce qu'on a fait dans cette Étape 1 (en 5 étapes faciles)

```
[ Données brutes Copernicus ]
           ↓
 1. Réduire la taille de la grille (2x2 pixels)
           ↓
 2. Garder UNIQUEMENT l'Algérie (retirer la mer et les pays voisins via dz.json)
           ↓
 3. Convertir les unités bizarres (Kelvin en °C, Joules en W/m², Pa en hPa)
           ↓
 4. Vérifier que les chiffres sont logiques (min, max, moyennes)
           ↓
 5. Mesurer le réchauffement climatique et les corrélations
```

---

## 📊 Explication détaillée des 5 Graphiques de votre Notebook

Voici comment commenter chaque graphique obtenu dans votre notebook `01_analyse_pretraitement.ipynb` :

---

### Graphique 1 : La ligne plate de valeurs manquantes (7.8%)
* **Ce qu'on voit :** Une ligne bleue complètement droite et horizontale fixée à **$0{,}078$ (7,8%)** de 2015 à 2026.
* **Ce que ça signifie :** 
  * Le taux de données manquantes ne change jamais au cours du temps : il ne dépend ni des saisons, ni des années.
  * Ces 7,8% représentent exactement **la mer Méditerranée** contenue dans le rectangle géographique de téléchargement (la mer a des valeurs `NaN` car ERA5-Land ne modélise que la terre).
* **Ce qu'il faut dire :** *« Cette ligne parfaitement stable prouve qu'il n'y a pas de pannes ou de trous temporels bizarres dans nos données : les seuls points manquants sont structurels (l'eau de mer), qu'on filtre ensuite facilement. »*

---

### Graphique 2 : Les histogrammes de distribution des 11 variables
* **Ce qu'on voit :**
  1. **`t2m` (Température à 2m) :** Elle a deux bosses (distribution bimodale) vers **$17^\circ\text{C}$** et **$34^\circ\text{C}$**. C'est le contraste très net entre l'hiver et l'été, mais aussi entre le nord plus tempéré et le sud saharien.
  2. **`tp` (Pluie) et `ro` (Ruissellement) :** Tout est écrasé à **0**. C'est normal : plus de 80% du territoire est le désert du Sahara, où il ne pleut quasiment jamais.
  3. **`e` (Évaporation) et `slhf` (Chaleur latente) :** Également concentrées vers 0 pour la même raison (sol sec sans eau à évaporer).
  4. **`sp` (Pression de surface) :** La majorité des points sont vers $950–1000\text{ hPa}$, mais la queue s'étire vers la gauche jusqu'à **$755\text{ hPa}$**, ce qui correspond aux zones de haute altitude (Atlas et Hoggar).
  5. **`u10` et `v10` (Vents) :** De belles cloches symétriques centrées sur $0\text{ m/s}$, avec des vents modérés.

---

### Graphique 3 : Les boxplots des cycles saisonniers (`t2m` et `str`)
* **Ce qu'on voit :**
  * **À gauche (`t2m`) :** Une forme sinusoïdale parfaite. Les températures sont au plus bas en janvier (médiane à $12^\circ\text{C}$, avec des gelées à $-3{,}4^\circ\text{C}$ en altitude) et atteignent leur pic en juillet/août (médiane à $34^\circ\text{C}$, avec des canicules à plus de $40^\circ\text{C}$).
  * **À droite (`str` - Rayonnement thermique net) :** La courbe est inversée. En été (juin/juillet), le sol surchauffé renvoie énormément de rayonnement infrarouge vers l'espace (valeurs très négatives, jusqu'à $-140\text{ W/m}^2$).

---

### Graphique 4 : La matrice de corrélation de Pearson (Heatmap rouge/bleu)
* **Ce qu'on voit :**
  * **$r = 1{,}00$ (rouge vif)** entre `slhf` et `e` : L'évaporation et le flux de chaleur latente sont physiquement la même chose (l'évaporation consomme de l'énergie thermique).
  * **$r = +0{,}54$ (rose)** entre `t2m` et `ssr` : Plus le soleil rayonne sur le sol, plus la température monte.
  * **$r = -0{,}57$ (bleu)** entre `t2m` et `str` : Lien direct entre chaleur et réémission thermique nocturne.
  * **$r = -0{,}80$ (bleu foncé)** entre `ssr` et `str` : L'équilibre radiatif entre le soleil reçu et l'énergie renvoyée.
* **Ce qu'il faut dire au prof :** *« Ces corrélations très fortes montrent une colinéarité importante. Un modèle linéaire simple deviendrait instable, ce qui justifie d'utiliser pour l'étape supervisée des arbres de décision (Random Forest) qui ne craignent pas la corrélation. »*

---

### Graphique 5 : La température annuelle & Tendance de réchauffement
* **Ce qu'on voit :** Les points bleus oscillent d'année en année (variabilité naturelle), mais la ligne pointillée rouge monte clairement de 2015 à 2026.
* **Le point clé à expliquer :**
  * La ligne de tendance brute monte à **$+0{,}151^\circ\text{C}/\text{an}$**.
  * **Pourquoi 2026 fait un saut à $25{,}65^\circ\text{C}$ ?** Parce que l'année 2026 n'est pas terminée (elle n'a que 9 mois dans le jeu) : il lui manque les mois froids d'octobre/novembre/décembre, donc sa moyenne paraît artificiellement plus chaude.
  * **Sur les années complètes (2015–2025) :** La hausse réelle est de **$+0{,}104^\circ\text{C}/\text{an}$**, ce qui représente **$+1{,}04^\circ\text{C}$ par décennie**.
  * C'est un réchauffement très rapide qui illustre le changement climatique sur l'Algérie.

---

## 🛠️ Rappel des choix techniques importants :

1. **Le groupement 2×2 (4 pixels) :**  
   Conforme à la précision de la prof (*« grouper 2x2=4 pixels ensemble »*), on passe de $0{,}1^\circ$ à $0{,}2^\circ$ pour diviser le volume de calcul par 4 sans écraser le relief.
2. **Le filtre frontière `dz.json` :**  
   Issu du lien SimpleMaps, il permet d'enlever les morceaux du Maroc, de la Tunisie et du Mali présents dans le rectangle Copernicus, pour ne garder **que l'Algérie**.
3. **Les conversions :**  
   Kelvin $\rightarrow$ °C, Joules $\rightarrow$ W/m², mètres $\rightarrow$ mm/j, Pascals $\rightarrow$ hPa pour que tout soit directement interprétable.

---

## 🗣️ Fiche antisèche : Réponses rapides aux questions de la prof

### Q1. « Pourquoi avez-vous fait un groupement 2×2 ? »
👉 *« C'est pour réduire le volume de données par 4 pour que les modèles tournent rapidement, tout en gardant une bonne précision sur le relief algérien sans lisser excessivement les montagnes. »*

### Q2. « Pourquoi avoir utilisé `dz.json` de SimpleMaps ? »
👉 *« Parce que Copernicus nous donne un rectangle qui déborde sur le Maroc, la Tunisie et le Mali. `dz.json` nous a permis de découper exactement les frontières de l'Algérie. »*

### Q3. « Comment avez-vous géré les valeurs manquantes ? »
👉 *« Les points en mer (100% de NaN) ont été supprimés. Et pour les quelques trous isolés sur terre, on a fait une interpolation linéaire temporelle sur le même point. »*

### Q4. « Pourquoi l'année 2026 saute tout en haut sur la courbe annuelle ? »
👉 *« Parce que l'année 2026 n'a que 9 mois. Il lui manque les mois froids d'hiver, ce qui tire sa moyenne vers le haut. Pour notre étude rigoureuse, on travaille sur les années complètes 2015–2025 (hausse de +0.104 °C/an). »*
