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


# --------------------------------------------------------------------------
def etape1(clean: pd.DataFrame) -> None:
    print("étape 1 — analyse et prétraitement")
    cols = [v for v in C.SHORT_NAMES if v in clean]

    save_table(pp.describe(clean).round(2).reset_index().rename(columns={"index": "variable"}),
               "01_statistiques_descriptives")

    rep = pp.missing_report(clean)
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, sub in rep.groupby("type"):
        ax.scatter(sub["longitude"], sub["latitude"], s=16, label=t)
    ax.set(xlabel="longitude", ylabel="latitude", title="Points de grille retenus après nettoyage")
    ax.legend()
    save_fig(fig, "01_masque_terre_mer")

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
              index=sp.FEATURES).sort_values().plot.barh(ax=ax)
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
