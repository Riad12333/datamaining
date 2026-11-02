"""Exporte les figures et tableaux du rapport depuis les données traitées.

    python -m scripts.export_assets

Écrit dans reports/figures/ (PNG) et reports/tables/ (CSV + Markdown).
Ne fabrique aucun chiffre : tout vient de data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clustering as cl              # noqa: E402
from src import frontiere as fr               # noqa: E402
from src import config as C                   # noqa: E402
from src import preprocessing as pp           # noqa: E402
from src import supervised as sp              # noqa: E402

sns.set_theme(style="whitegrid")
FIG = C.ROOT / "reports" / "figures"
TAB = C.ROOT / "reports" / "tables"


def save_fig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  reports/figures/{name}.png")


def save_table(df: pd.DataFrame, name: str, **kw) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=kw.get("index", False))
    (TAB / f"{name}.md").write_text(df.to_markdown(index=kw.get("index", False)))
    print(f"  tableau reports/tables/{name}.csv|.md")


LIBELLES = {
    "t2m": "température à 2 m (°C)", "d2m": "point de rosée (°C)",
    "slhf": "chaleur latente (W m⁻²)", "ssr": "rayonnement solaire net (W m⁻²)",
    "str": "rayonnement thermique net (W m⁻²)", "ro": "ruissellement (mm/j)",
    "e": "évaporation (mm/j)", "u10": "vent zonal (m s⁻¹)",
    "v10": "vent méridien (m s⁻¹)", "sp": "pression de surface (hPa)",
    "tp": "précipitations (mm/j)", "sshf_bilan": "chaleur sensible (W m⁻²)",
}


def grille_2d(df: pd.DataFrame, col: str):
    """Repasse une colonne du format long au format carte (lat × lon)."""
    g = df.pivot_table(index="latitude", columns="longitude", values=col)
    return g.columns.to_numpy(), g.index.to_numpy(), g.to_numpy()


def carte(ax, df, col, titre, cmap="viridis", contour=True, **kw):
    """Trace une carte d'une variable, avec la frontière algérienne par-dessus."""
    from shapely.geometry import MultiPolygon

    lon, lat, z = grille_2d(df, col)
    m = ax.pcolormesh(lon, lat, z, cmap=cmap, shading="nearest", **kw)
    plt.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
    if contour:
        geom = fr.telecharger_contour()
        for poly in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]):
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", lw=0.7)
    ax.set(title=titre, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    return m


# --------------------------------------------------------------------------
# Puce 2 — Data Preprocessing
# --------------------------------------------------------------------------
def figure_agregation() -> None:
    """Effet visuel de l'agrégation 2×2 : avant / après, à l'identique."""
    ds = pp.open_raw()
    juillet = ds.sel(time="2015-07-01")
    reduit = pp.coarsen(ds).sel(time="2015-07-01")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (d, lab) in zip(axes, [(juillet, f"{C.NATIVE_GRID}° natif"),
                                   (reduit, f"{C.GRID}° après agrégation "
                                            f"{C.COARSEN_FACTOR}×{C.COARSEN_FACTOR}")]):
        z = d["t2m"].values - 273.15
        m = ax.pcolormesh(d.longitude, d.latitude, z, cmap="inferno", shading="nearest",
                          vmin=20, vmax=45)
        plt.colorbar(m, ax=ax, fraction=0.046, pad=0.03, label="°C")
        n = d.latitude.size * d.longitude.size
        ax.set(title=f"t2m juillet 2015 — {lab}\n{d.latitude.size}×{d.longitude.size} "
                     f"= {n:,} points".replace(",", " "), aspect="equal")
    fig.suptitle("Réduction de la résolution spatiale : 4 pixels regroupés par maille")
    fig.tight_layout()
    save_fig(fig, "01_prep_effet_agregation")


def figure_manquants(clean: pd.DataFrame) -> None:
    """Part de valeurs manquantes dans le brut, avant nettoyage."""
    brut = pp.to_long(pp.coarsen(pp.open_raw()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    part = brut.set_index("time")["t2m"].isna().groupby(level=0).mean()
    axes[0].plot(part.index, part.values * 100, lw=1.2)
    axes[0].set(title="Part de `t2m` manquante, mois par mois (grille brute)",
                ylabel="%", xlabel="")
    axes[0].set_ylim(0, max(part.max() * 110, 1))

    rep = pp.missing_report(brut)
    axes[1].hist(rep["nan_ratio"] * 100, bins=50, color="#546e7a")
    axes[1].set(title="Taux de NaN par point de grille", xlabel="% de mois manquants",
                ylabel="nombre de points", yscale="log")
    axes[1].axvline(99, color="crimson", ls="--", label="seuil « hors domaine » (99 %)")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "01_prep_valeurs_manquantes")


# --------------------------------------------------------------------------
# Puce 1 — Exploratory Data Analysis
# --------------------------------------------------------------------------
def figure_climatologie(clean: pd.DataFrame) -> None:
    """Cartes climatologiques : la photographie météo de la zone d'étude.

    Restreint aux années complètes : sinon janvier serait moyenné sur 12 ans et
    décembre sur 11, ce qui biaiserait la comparaison entre mois.
    """
    clean = pp.complete_years(clean)
    moy = clean.groupby(["latitude", "longitude"]).mean(numeric_only=True).reset_index()
    jan = clean[clean.month == 1].groupby(["latitude", "longitude"]).mean(numeric_only=True).reset_index()
    jul = clean[clean.month == 7].groupby(["latitude", "longitude"]).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    carte(axes[0, 0], jan, "t2m", "t2m — janvier", cmap="RdYlBu_r", vmin=0, vmax=40)
    carte(axes[0, 1], jul, "t2m", "t2m — juillet", cmap="RdYlBu_r", vmin=0, vmax=40)
    carte(axes[0, 2], moy, "t2m", "t2m — moyenne annuelle", cmap="RdYlBu_r", vmin=0, vmax=40)
    carte(axes[1, 0], moy, "tp", "précipitations (mm/j)", cmap="YlGnBu")
    carte(axes[1, 1], moy, "ssr", "rayonnement solaire net (W m⁻²)", cmap="plasma")
    carte(axes[1, 2], moy, "sp", "pression de surface (hPa)\n(proxy du relief)", cmap="terrain_r")
    fig.suptitle(f"Climatologie {clean.year.min()}–{clean.year.max()} — moyennes par point de grille",
                 y=1.00)
    fig.tight_layout()
    save_fig(fig, "01_eda_cartes_climatologie")


def figure_vent_et_contrastes(clean: pd.DataFrame) -> None:
    """Vent moyen (quiver) et amplitude saisonnière, sur années complètes."""
    clean = pp.complete_years(clean)
    moy = clean.groupby(["latitude", "longitude"]).mean(numeric_only=True).reset_index()
    moy["vitesse"] = np.hypot(moy["u10"], moy["v10"])

    jan = clean[clean.month == 1].groupby(["latitude", "longitude"])["t2m"].mean()
    jul = clean[clean.month == 7].groupby(["latitude", "longitude"])["t2m"].mean()
    amp = (jul - jan).rename("amplitude").reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    carte(axes[0], moy, "vitesse", "Vent moyen à 10 m (m s⁻¹)", cmap="BuPu")
    pas = 6                                   # une flèche sur six, sinon illisible
    sub = moy.iloc[::pas]
    axes[0].quiver(sub["longitude"], sub["latitude"], sub["u10"], sub["v10"],
                   color="black", scale=40, width=0.003)
    carte(axes[1], amp, "amplitude", "Amplitude saisonnière t2m (juillet − janvier, °C)",
          cmap="magma")
    fig.tight_layout()
    save_fig(fig, "01_eda_vent_amplitude")


def figure_tendance_spatiale(clean: pd.DataFrame) -> None:
    """Tendance de t2m par point de grille, sur les années complètes."""
    comp = pp.complete_years(clean)
    annuel = comp.groupby(["latitude", "longitude", "year"])["t2m"].mean().reset_index()
    annees = np.sort(annuel["year"].unique()).astype(float)
    xc = annees - annees.mean()

    wide = annuel.pivot_table(index=["latitude", "longitude"], columns="year", values="t2m")
    bloc = wide.to_numpy()
    pentes = ((bloc - bloc.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc ** 2).sum()
    tend = wide.index.to_frame(index=False)
    tend["tendance"] = pentes

    lim = float(np.nanpercentile(np.abs(pentes), 98))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    carte(ax, tend, "tendance", f"Tendance de t2m par point, {int(annees[0])}–{int(annees[-1])}"
                                f"\n(°C/an ; moyenne {pentes.mean():+.3f})",
          cmap="RdBu_r", vmin=-lim, vmax=lim)
    save_fig(fig, "01_eda_carte_tendance")
    print(f"  [chiffre clé] tendance locale : min {pentes.min():+.3f}, "
          f"max {pentes.max():+.3f} °C/an")


# --------------------------------------------------------------------------
# Puce 3 — Feature Engineering
# --------------------------------------------------------------------------
def figures_feature_engineering(clean: pd.DataFrame) -> None:
    """Livrables visuels des deux branches de feature engineering."""
    train, test, edges = sp.prepare(clean, verbose=False)
    # `edges` porte ±inf aux extrémités (pour absorber le test hors étendue) :
    # on retrouve les bornes réelles depuis l'étendue du train.
    interieures = [b for b in edges if np.isfinite(b)]
    lo, hi = float(train["t2m_target"].min()), float(train["t2m_target"].max())
    bornes = [lo] + interieures + [hi]

    # --- branche supervisée : la discrétisation équi-largeur ---------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(train["t2m_target"], bins=80, color="#90a4ae")
    for b in interieures:
        axes[0].axvline(b, color="crimson", ls="--")
    for lab, a, b in zip(C.BIN_LABELS, bornes[:-1], bornes[1:]):
        axes[0].text((a + b) / 2, axes[0].get_ylim()[1] * 0.92, lab, ha="center",
                     fontsize=9, bbox=dict(fc="white", ec="crimson", alpha=0.85))
    axes[0].set(title=f"Cible t2m(t+1) et bornes équi-largeur (train)\n"
                      f"largeur de chaque intervalle : {(hi - lo) / C.N_BINS:.2f} °C",
                xlabel="°C")

    parts = pd.concat([
        train["classe"].value_counts(normalize=True).rename("train"),
        test["classe"].value_counts(normalize=True).rename("test")], axis=1
    ).reindex(C.BIN_LABELS)
    parts.plot.bar(ax=axes[1], color=["#1565c0", "#ef6c00"], rot=0)
    axes[1].set(title="Répartition des classes — déséquilibre dû à l'équi-largeur",
                ylabel="part")
    for cont in axes[1].containers:
        axes[1].bar_label(cont, fmt="%.1f%%",
                          labels=[f"{v*100:.1f}%" for v in cont.datavalues])
    fig.tight_layout()
    save_fig(fig, "01_fe_sup_discretisation")

    # --- branche supervisée : où tombe chaque classe ----------------------
    # `classe` est catégorielle : pandas refuse d'agréger ce dtype, on repasse
    # par le code entier puis par la valeur la plus fréquente.
    codes = {c: i for i, c in enumerate(C.BIN_LABELS)}
    tmp = train.assign(code=train["classe"].map(codes).astype(float))
    modale = (tmp.groupby(["latitude", "longitude"])["code"]
              .agg(lambda s: s.value_counts().idxmax()).reset_index())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    carte(ax, modale, "code", "Classe majoritaire par point de grille (train)",
          cmap=plt.matplotlib.colors.ListedColormap(["#2196f3", "#ffb300", "#d32f2f"]),
          vmin=-0.5, vmax=2.5)
    save_fig(fig, "01_fe_sup_carte_classes")

    # --- branche non supervisée : profils et signal climatique ------------
    prof = cl.build_year_profiles(clean)
    mois_t2m = [f"t2m_{m:02d}" for m in range(1, 13)]
    mois_str = [f"str_{m:02d}" for m in range(1, 13)]

    repere = {"côte (Alger)": (36.7, 3.1), "hauts plateaux (Sétif)": (36.2, 5.4),
              "Sahara (Adrar)": (27.9, -0.3), "Hoggar (Tamanrasset)": (22.8, 5.5)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for nom, (la, lo) in repere.items():
        d = prof.assign(dist=(prof.latitude - la).abs() + (prof.longitude - lo).abs())
        p = d.nsmallest(1, "dist").iloc[0]
        axes[0].plot(range(1, 13), [p[c] for c in mois_t2m], marker="o", label=nom)
        axes[1].plot(range(1, 13), [p[c] for c in mois_str], marker="o", label=nom)
    axes[0].set(title="Profil annuel de t2m (12 colonnes de l'instance)",
                xlabel="mois", ylabel="°C")
    axes[1].set(title="Profil annuel du rayonnement thermique net",
                xlabel="mois", ylabel="W m⁻²")
    axes[0].legend(fontsize=8)

    anom = prof.groupby("year")[["t2m_anomalie", "str_anomalie"]].mean()
    anom.plot(ax=axes[2], marker="o")
    axes[2].axhline(0, color="grey", lw=1)
    axes[2].set(title="Features de tendance long terme :\nanomalie moyenne par année",
                xlabel="année")
    fig.tight_layout()
    save_fig(fig, "01_fe_nonsup_profils")

    derive = prof.drop_duplicates(["latitude", "longitude"])
    lim = float(np.nanpercentile(derive["t2m_derive"].abs(), 98))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    carte(ax, derive, "t2m_derive",
          "Feature `t2m_derive` : dérive pluriannuelle par point (°C/an)",
          cmap="RdBu_r", vmin=-lim, vmax=lim)
    save_fig(fig, "01_fe_nonsup_carte_derive")


# --------------------------------------------------------------------------
def etape1(clean: pd.DataFrame) -> None:
    print("étape 1 — analyse et prétraitement")
    cols = [v for v in C.SHORT_NAMES if v in clean]

    save_table(pp.describe(clean).round(2).reset_index().rename(columns={"index": "variable"}),
               "01_statistiques_descriptives")

    print(" puce 2 — Data Preprocessing")
    figure_masque(clean)
    figure_agregation()
    figure_manquants(clean)

    print(" puce 1 — Exploratory Data Analysis")
    figure_climatologie(clean)
    figure_vent_et_contrastes(clean)
    figure_tendance_spatiale(clean)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, v in zip(axes.ravel(), cols):
        sns.histplot(clean[v], bins=60, ax=ax)
        ax.set(title=v, xlabel="")
    fig.suptitle("Distribution des 12 variables après conversion d'unités", y=1.01)
    fig.tight_layout()
    save_fig(fig, "01_distributions")

    fig, ax = plt.subplots(figsize=(9, 7))
    corr = pp.correlations(clean)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Corrélations de Pearson")
    save_fig(fig, "01_correlations")
    save_table(corr.round(3).reset_index().rename(columns={"index": "variable"}), "01_correlations")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(data=clean, x="month", y="t2m", ax=axes[0]).set(title="Cycle saisonnier — t2m (°C)")
    # La tendance se calcule sur les années complètes : une année tronquée
    # (ex. janvier-septembre) a une moyenne biaisée vers le haut.
    annuel = pp.complete_years(clean).groupby("year")["t2m"].mean()
    annees = annuel.index.to_numpy(float)
    pente = np.polyfit(annees, annuel.values, 1)
    axes[1].plot(annees, annuel.values, marker="o")
    axes[1].plot(annees, np.poly1d(pente)(annees), "--", color="crimson",
                 label=f"tendance : {pente[0]:+.3f} °C/an")
    axes[1].set(title=f"Température moyenne annuelle (°C) — années complètes "
                      f"{int(annees[0])}-{int(annees[-1])}", xlabel="année")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "01_saisonnalite_tendance")
    print(f"  [chiffre clé] tendance de t2m : {pente[0]:+.4f} °C/an")

    print(" puce 3 — Feature Engineering")
    figures_feature_engineering(clean)


def figure_masque(clean: pd.DataFrame) -> None:
    """Grille complète : mer, pays voisins, mailles frontalières, Algérie retenue.

    Montrer ce qui est écarté est plus informatif que de ne montrer que ce qui
    est gardé : c'est la justification visuelle du découpage sur le contour.
    """
    from shapely.geometry import MultiPolygon

    brut = pp.to_long(pp.coarsen(pp.open_raw()))
    rep = pp.missing_report(brut)
    terre = rep[rep["type"] != "hors_terre"][["latitude", "longitude"]].reset_index(drop=True)
    mer = rep[rep["type"] == "hors_terre"]
    zones = fr.classer(terre)

    fig, ax = plt.subplots(figsize=(9, 6))
    contour = fr.telecharger_contour()
    polys = contour.geoms if isinstance(contour, MultiPolygon) else [contour]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", lw=1.1, zorder=5, label="frontière algérienne")

    styles = {
        "hors_terre": ("#cfd8dc", "mer / hors ERA5-Land (écarté)"),
        "hors_algerie": ("#ef9a9a", "pays voisins (écarté)"),
        "frontalier": ("#ffb74d", f"maille frontalière (gardée, buffer {C.FRONTIERE_BUFFER}°)"),
        "algerie": ("#1565c0", "Algérie (gardé)"),
    }
    ax.scatter(mer["longitude"], mer["latitude"], s=10, c=styles["hors_terre"][0],
               label=styles["hors_terre"][1])
    for z in ("hors_algerie", "frontalier", "algerie"):
        sub = zones[zones["zone"] == z]
        c, lab = styles[z]
        ax.scatter(sub["longitude"], sub["latitude"], s=14, c=c, label=f"{lab} — {len(sub)}")

    ax.set(xlabel="longitude", ylabel="latitude",
           title="Points de grille : sélection de la zone d'étude")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_aspect("equal")
    save_fig(fig, "01_masque_terre_mer")

    tab = zones["zone"].value_counts().rename("points").reset_index()
    tab.columns = ["zone", "points"]
    tab.loc[len(tab)] = ["mer_hors_era5land", len(mer)]
    tab["part"] = (tab["points"] / tab["points"].sum()).round(3)
    save_table(tab, "01_selection_zone")


def etape2(clean: pd.DataFrame):
    print("étape 2 — supervisé")
    train, test, edges = sp.prepare(clean, verbose=False)
    table, fitted, reports = sp.evaluate(train, test)
    save_table(table.round(4), "02_comparaison_modeles")

    bornes = pd.DataFrame({
        "intervalle": C.BIN_LABELS,
        "borne_inf_C": [round(float(b), 2) for b in edges[:-1]],
        "borne_sup_C": [round(float(b), 2) for b in edges[1:]],
        "part_train": train["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS).round(3).values,
        "part_test": test["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS).round(3).values,
    })
    save_table(bornes, "02_bornes_et_classes")

    fig, ax = plt.subplots(figsize=(9, 4))
    table.set_index("modele")[["accuracy", "f1_macro", "kappa"]].plot.bar(ax=ax)
    ax.set(title="Comparaison des modèles (jeu de test)", ylim=(0, 1))
    save_fig(fig, "02_comparaison_modeles")

    best = table.loc[0, "modele"]
    cm = pd.DataFrame(reports[best]["confusion"], index=C.BIN_LABELS, columns=C.BIN_LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm / cm.sum(axis=1).values[:, None], annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set(xlabel="prédit", ylabel="réel", title=f"Matrice de confusion normalisée — {best}")
    save_fig(fig, "02_matrice_confusion")
    save_table(cm.reset_index().rename(columns={"index": "reel"}), "02_matrice_confusion")

    fig, ax = plt.subplots(figsize=(7, 5))
    pd.Series(fitted["random_forest"].feature_importances_,
              index=sp.features_for(train)).sort_values().plot.barh(ax=ax)
    ax.set_title("Importance des variables (forêt aléatoire)")
    save_fig(fig, "02_importance_variables")
    print(f"  [chiffre clé] meilleur modèle : {best} "
          f"(F1 macro {table.loc[0, 'f1_macro']:.3f}, accuracy {table.loc[0, 'accuracy']:.3f})")
    return table


def etape3(clean: pd.DataFrame):
    print("étape 3 — clustering")
    profiles = cl.build_year_profiles(clean)

    scores_k = cl.elbow(profiles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    scores_k.plot(x="k", y="inertie", marker="o", ax=axes[0], title="Coude (inertie)")
    scores_k.plot(x="k", y="silhouette", marker="o", ax=axes[1], title="Silhouette")
    fig.tight_layout()
    save_fig(fig, "03_choix_k")
    save_table(scores_k.round(4), "03_choix_k")

    labels, scores = cl.run_all_years(profiles)
    resume = (scores.groupby("algo")[["n_clusters", "bruit", "silhouette",
                                      "calinski", "davies_bouldin"]]
              .mean().round(3).reset_index())
    save_table(resume, "03_comparaison_algorithmes")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m, titre in zip(axes, ["silhouette", "calinski", "davies_bouldin"],
                            ["Silhouette (↑)", "Calinski-Harabasz (↑)", "Davies-Bouldin (↓)"]):
        sns.lineplot(data=scores, x="year", y=m, hue="algo", marker="o", ax=ax)
        ax.set_title(titre)
    fig.tight_layout()
    save_fig(fig, "03_metriques_par_annee")

    derniere = labels["year"].max()
    sub = labels[labels["year"] == derniere]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, algo in zip(axes.ravel(), ["kmeans", "agnes", "dbscan", "ae_kmeans"]):
        sns.scatterplot(data=sub, x="longitude", y="latitude", hue=algo,
                        palette="tab10", s=22, ax=ax)
        ax.set_title(f"{algo} — {derniere}")
    fig.tight_layout()
    save_fig(fig, "03_cartes_clusters")

    merged = labels.merge(profiles, on=cl.KEYS)
    mois = [f"t2m_{m:02d}" for m in range(1, 13)]
    prof = merged[merged["kmeans"] >= 0].groupby("kmeans")[mois].mean().T
    prof.index = range(1, 13)
    fig, ax = plt.subplots(figsize=(9, 4))
    prof.plot(marker="o", ax=ax, xlabel="mois", ylabel="t2m (°C)",
              title="Profil annuel moyen par cluster (K-Means)")
    save_fig(fig, "03_profils_clusters")

    evo = cl.evolution(labels, profiles, algo="kmeans")
    save_table(evo.round(3), "03_evolution_clusters")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    evo.pivot(index="year", columns="cluster", values="part").plot.area(
        ax=axes[0], title="Part des points par cluster")
    evo.pivot(index="year", columns="cluster", values="t2m_moyen").plot(
        marker="o", ax=axes[1], title="Température moyenne par cluster (°C)")
    evo.pivot(index="year", columns="cluster", values="anomalie").plot(
        marker="o", ax=axes[2], title="Anomalie moyenne par cluster (°C)")
    axes[2].axhline(0, color="grey", lw=1)
    fig.tight_layout()
    save_fig(fig, "03_evolution_clusters")

    trans = cl.transitions(labels, algo="kmeans")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(trans, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(f"Transitions {labels['year'].min()} → {labels['year'].max()} (K-Means)")
    save_fig(fig, "03_transitions")
    save_table(trans.reset_index(), "03_transitions")


def provenance(clean: pd.DataFrame) -> str:
    """Trace l'origine des données : synthétique ou CDS réel."""
    import xarray as xr

    src = "inconnue (NetCDF brut absent)"
    if C.FILE_RAW.exists():
        attrs = xr.open_dataset(C.FILE_RAW).attrs
        src = attrs.get("source", "Copernicus CDS (attribut source absent)")
    return (f"source : {src}\n"
            f"période : {clean['year'].min()}–{clean['year'].max()}\n"
            f"points de grille : {clean.groupby(['latitude', 'longitude']).ngroups:,}\n"
            f"lignes : {len(clean):,}")


if __name__ == "__main__":
    if not C.FILE_CLEAN.exists():
        raise SystemExit("data/processed/era5_clean.parquet absent — "
                         "lancer d'abord src.download puis src.preprocessing")
    clean = pd.read_parquet(C.FILE_CLEAN)

    etape1(clean)
    etape2(clean)
    etape3(clean)

    info = provenance(clean)
    (C.ROOT / "reports" / "PROVENANCE.txt").write_text(info + "\n")
    print("\n=== provenance des résultats ===")
    print(info)
