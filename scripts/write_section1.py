"""Rédige la section « Étape 1 — Analyse et prétraitement » du rapport.

    python -m scripts.write_section1

Écrit reports/sections/01_analyse_pretraitement.md. Tous les chiffres sont
calculés depuis les données présentes : relancer le script après avoir changé
de jeu de données met la section à jour, sans risque de chiffre périmé.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C            # noqa: E402
from src import preprocessing as pp    # noqa: E402

OUT = C.ROOT / "reports" / "sections" / "01_analyse_pretraitement.md"

LIBELLES = {
    "t2m": "température à 2 m", "d2m": "point de rosée à 2 m",
    "slhf": "flux de chaleur latente", "ssr": "rayonnement solaire net",
    "str": "rayonnement thermique net", "sshf": "flux de chaleur sensible",
    "ro": "ruissellement", "e": "évaporation totale",
    "u10": "vent zonal à 10 m", "v10": "vent méridien à 10 m",
    "sp": "pression de surface", "tp": "précipitations totales",
}
UNITES = {"t2m": "°C", "d2m": "°C", "slhf": "W m⁻²", "ssr": "W m⁻²",
          "str": "W m⁻²", "sshf": "W m⁻²", "ro": "mm/j", "e": "mm/j",
          "u10": "m s⁻¹", "v10": "m s⁻¹", "sp": "hPa", "tp": "mm/j"}
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]


def fr(n) -> str:
    """Entier avec espace insécable fine comme séparateur de milliers."""
    return f"{n:,}".replace(",", "\u202f")


def grille_info() -> dict:
    """Taille de grille avant et après agrégation, lue sur le NetCDF brut."""
    if not C.FILE_RAW.exists():
        return {}
    import xarray as xr

    ds = xr.open_dataset(C.FILE_RAW)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    res_av = float(abs(ds["latitude"].diff("latitude").mean()))
    nlat, nlon = ds["latitude"].size, ds["longitude"].size
    red = ds.coarsen(latitude=C.COARSEN_FACTOR, longitude=C.COARSEN_FACTOR,
                     boundary="trim").mean()
    return {
        "res_avant": res_av,
        "res_apres": res_av * C.COARSEN_FACTOR,
        "n_avant": nlat * nlon,
        "n_apres": red["latitude"].size * red["longitude"].size,
        "forme_avant": f"{nlat}×{nlon}",
        "forme_apres": f"{red['latitude'].size}×{red['longitude'].size}",
        "source": ds.attrs.get("source", "Copernicus CDS"),
        "nan_brut": float(np.isnan(ds["t2m"].values).mean()),
    }


def correlations_notables(corr: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Les n paires de variables les plus corrélées, en valeur absolue."""
    m = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    top = m.reindex(m.abs().sort_values(ascending=False).index).head(n)
    return pd.DataFrame({
        "variable A": [f"`{a}` ({LIBELLES.get(a, a)})" for a, _ in top.index],
        "variable B": [f"`{b}` ({LIBELLES.get(b, b)})" for _, b in top.index],
        "r": [round(v, 3) for v in top.values],
    })


def main() -> None:
    if not C.FILE_CLEAN.exists():
        raise SystemExit("data/processed/era5_clean.parquet absent — "
                         "lancer src.download puis src.preprocessing")

    clean = pd.read_parquet(C.FILE_CLEAN)
    g = grille_info()
    synthetique = "substitution" in g.get("source", "")

    # --- chiffres ---------------------------------------------------------
    n_points = clean.groupby(["latitude", "longitude"]).ngroups
    n_mois = clean["time"].nunique()
    y0, y1 = int(clean["year"].min()), int(clean["year"].max())

    stats = pp.describe(clean).round(2)
    stats.insert(0, "unité", [UNITES.get(v, "") for v in stats.index])
    stats.insert(0, "variable", [f"`{v}` — {LIBELLES.get(v, v)}" for v in stats.index])
    stats = stats[["variable", "unité", "mean", "std", "min", "50%", "max"]]
    stats.columns = ["variable", "unité", "moyenne", "écart-type", "min", "médiane", "max"]

    corr = pp.correlations(clean)
    notables = correlations_notables(corr)
    corr_t2m = corr["t2m"].drop("t2m").sort_values(key=abs, ascending=False)

    cycle = clean.groupby("month")["t2m"].mean()
    amplitude = float(cycle.max() - cycle.min())
    completes = pp.complete_years(clean)
    annees_partielles = sorted(set(clean["year"]) - set(completes["year"]))
    annuel = completes.groupby("year")["t2m"].mean()
    annees = annuel.index.to_numpy(dtype=float)
    pente, ordonnee = np.polyfit(annees, annuel.values, 1)
    # Incertitude de la pente : écart-type des résidus / sqrt(Sxx)
    resid = annuel.values - (pente * annees + ordonnee)
    sxx = float(((annees - annees.mean()) ** 2).sum())
    err_pente = float(np.std(resid, ddof=2) / np.sqrt(sxx))

    lat_corr = float(corr.loc["t2m", "latitude"]) if "latitude" in corr else np.nan

    # --- rédaction --------------------------------------------------------
    bandeau = (
        "> **⚠ Données de substitution.** Les chiffres de cette section "
        f"proviennent de `{g.get('source', 'inconnue')}` et **ne sont pas des "
        "résultats ERA5**. Relancer `src.download` (avec une clé CDS) puis "
        "`scripts.write_section1` pour les remplacer.\n"
        if synthetique else
        "> Chiffres calculés sur ERA5-Land, régénérés par "
        "`python -m scripts.write_section1`.\n"
    )

    parties = [
        "## Étape 1 — Analyse et prétraitement\n",
        bandeau,
        "### 1.1 Jeu de données\n",
        f"Les données proviennent de la réanalyse **ERA5-Land monthly averaged** "
        f"(Copernicus Climate Data Store), sur l'emprise `{C.AREA}` "
        f"(Nord, Ouest, Sud, Est) couvrant l'Algérie et une marge maritime. "
        f"La période retenue va de **{y0} à {y1}**, soit **{n_mois} pas de temps "
        f"mensuels**. Douze variables sont extraites : les onze variables "
        f"explicatives imposées par l'énoncé et la cible `t2m`.\n",
        "La marge maritime est volontaire : elle produit les valeurs manquantes "
        "que la section 1.3 traite.\n",
    ]

    if g:
        parties += [
            "### 1.2 Réduction de la résolution spatiale\n",
            f"L'énoncé impose de réduire la résolution spatiale d'un facteur "
            f"quatre en agrégeant les points voisins, soit **{C.COARSEN_FACTOR ** 2} "
            f"pixels regroupés** par maille. "
            f"L'agrégation est réalisée par moyenne de blocs "
            f"{C.COARSEN_FACTOR}×{C.COARSEN_FACTOR} "
            f"(`xarray.Dataset.coarsen`), qui fait passer la grille de "
            f"**{g['res_avant']:.2f}° à {g['res_apres']:.2f}°**, soit de "
            f"**{g['forme_avant']} = {fr(g['n_avant'])} points** à "
            f"**{g['forme_apres']} = {fr(g['n_apres'])} points** "
            f"(facteur {g['n_avant'] / g['n_apres']:.1f}).\n",
            "Deux choix techniques méritent justification.\n",
            "`boundary=\"trim\"` écarte les lignes et colonnes surnuméraires "
            "plutôt que de former un bloc partiel en bordure : un bloc incomplet "
            f"n'a pas la même variance d'échantillonnage qu'un bloc de "
            f"{C.COARSEN_FACTOR ** 2} points, et le conserver introduirait un "
            "biais sur le pourtour du domaine.\n",
            "La moyenne ignore les valeurs manquantes. Un bloc à cheval sur la "
            "côte conserve donc la moyenne de ses seuls points terrestres au lieu "
            "d'être annulé par un unique point marin ; un bloc entièrement marin "
            "reste manquant et sera écarté à l'étape suivante.\n",
            "**Compromis assumé.** L'agrégation divise le nombre de points par "
            f"{C.COARSEN_FACTOR ** 2} et lisse le bruit local, au prix d'une perte "
            f"de détail orographique : à {g['res_apres']:.1f}°, le contraste entre "
            "l'Atlas tellien et les hauts plateaux est fortement atténué.\n",
        ]

    rep_txt = ""
    if g:
        rep_txt = (f" Sur la grille brute, {g['nan_brut']:.1%} des valeurs de `t2m` "
                   f"sont manquantes.")

    parties += [
        "### 1.3 Traitement des valeurs manquantes\n",
        "ERA5-Land n'est défini que sur les terres émergées : les points marins "
        f"sont manquants sur toute la période.{rep_txt} Deux régimes sont "
        "distingués, car ils n'appellent pas le même traitement.\n",
        "| régime | critère | traitement | justification |\n"
        "|---|---|---|---|\n"
        "| hors domaine | > 99 % de NaN sur `t2m` | point supprimé | aucune "
        "information à récupérer : le point n'est jamais observé |\n"
        "| trou isolé | série par ailleurs complète | interpolation linéaire "
        "temporelle, **au sein du point** | la série mensuelle d'un point est "
        "lisse ; interpoler entre deux mois voisins est plus fidèle que supprimer "
        "l'instance |\n",
        f"Après nettoyage, **{fr(n_points)} points de grille** sont retenus, pour "
        f"**{fr(len(clean))} lignes** ({fr(n_points)} × {n_mois}).\n",
        "**Figure 1** — `reports/figures/01_masque_terre_mer.png` : répartition "
        "spatiale des points conservés et écartés.\n",
        "### 1.4 Conversion des unités\n",
        "Les conversions servent l'interprétabilité, non la performance : les "
        "modèles utilisés sont insensibles à un changement d'échelle affine. "
        "Elles sont implémentées de façon **idempotente** (test préalable sur la "
        "médiane), de sorte qu'une seconde exécution du script ne convertisse pas "
        "deux fois.\n",
        "| variables | de | vers | opération |\n|---|---|---|---|\n"
        f"| {', '.join(f'`{v}`' for v in C.KELVIN_VARS)} | K | °C | − 273,15 |\n"
        f"| {', '.join(f'`{v}`' for v in C.FLUX_VARS)} | J m⁻² (cumul journalier) "
        f"| W m⁻² | ÷ {fr(C.SECONDS_PER_DAY)} |\n" +
        f"| {', '.join(f'`{v}`' for v in C.DEPTH_VARS)} | m | mm/jour | × 1000 |\n"
        f"| {', '.join(f'`{v}`' for v in C.PRESSURE_VARS)} | Pa | hPa | ÷ 100 |\n",
        "Le facteur 86 400 appliqué aux flux mérite un mot : ERA5-Land fournit "
        "pour les moyennes mensuelles un **cumul journalier moyen** en J m⁻², que "
        "l'on ramène à une puissance moyenne en W m⁻² en divisant par le nombre de "
        "secondes dans une journée.\n",
        "### 1.5 Statistiques descriptives\n",
        stats.to_markdown(index=False) + "\n",
        "**Figure 2** — `reports/figures/01_distributions.png` : histogrammes des "
        "douze variables après conversion.\n",
        "### 1.6 Corrélations\n",
        "**Figure 3** — `reports/figures/01_correlations.png` : matrice de "
        "corrélation de Pearson.\n",
        "Les associations les plus fortes sont les suivantes.\n",
        notables.to_markdown(index=False) + "\n",
        "Vis-à-vis de la cible `t2m`, les trois variables les plus informatives "
        "sont " + ", ".join(
            f"`{v}` (r = {r:+.2f})" for v, r in corr_t2m.head(3).items()
        ) + ".\n",
        "**Conséquence pour la modélisation.** Ces corrélations élevées signalent "
        "une forte **colinéarité**. Elle pénaliserait un modèle linéaire, dont les "
        "coefficients deviendraient instables et ininterprétables, mais reste sans "
        "effet sur les arbres et les forêts, qui sélectionnent une variable à la "
        "fois. Elle explique aussi pourquoi l'importance des variables d'une forêt "
        "aléatoire doit être lue avec prudence : deux variables redondantes se "
        "partagent l'importance au lieu de la cumuler.\n",
        "### 1.7 Saisonnalité et tendance\n",
        "**Figure 4** — `reports/figures/01_saisonnalite_tendance.png` : cycle "
        "saisonnier mensuel et moyenne annuelle.\n",
        f"Le cycle saisonnier domine la variabilité : la température moyenne passe "
        f"de **{cycle.min():.1f} °C** en {MOIS[cycle.idxmin() - 1]} à "
        f"**{cycle.max():.1f} °C** en {MOIS[cycle.idxmax() - 1]}, soit une amplitude de "
        f"**{amplitude:.1f} °C**.\n",
        (f"Les années {annees_partielles} sont incomplètes dans le jeu téléchargé "
         f"et sont **écartées du calcul de tendance** : une année tronquée aux "
         f"premiers mois a une moyenne biaisée vers le haut, ce qui gonflerait "
         f"artificiellement la pente.\n"
         if annees_partielles else ""),
        f"La régression linéaire des moyennes annuelles sur {len(annees):.0f} années "
        f"donne une tendance de **{pente:+.4f} ± {err_pente:.4f} °C/an**, soit "
        f"**{pente * 10:+.2f} °C par décennie**.\n",
        f"Cette tendance est à manier avec précaution : {len(annees):.0f} années "
        "constituent une série courte au regard de la variabilité interannuelle, "
        "et l'incertitude ci-dessus ne tient pas compte de l'autocorrélation "
        "temporelle, qui la sous-estime.\n",
    ]

    if not np.isnan(lat_corr):
        parties.append(
            f"La structure spatiale est dominée par le gradient nord-sud "
            f"(r = {lat_corr:+.2f} entre `t2m` et la latitude) : c'est le facteur "
            f"qui structurera les clusters de l'étape 3.\n")

    parties += [
        "### 1.8 Sorties produites\n",
        "| fichier | contenu |\n|---|---|\n"
        f"| `{C.FILE_CLEAN.relative_to(C.ROOT)}` | table nettoyée : "
        f"{fr(len(clean))} lignes × {clean.shape[1]} colonnes |\n" +
        "| `reports/figures/01_*.png` | les quatre figures ci-dessus |\n"
        "| `reports/tables/01_*.csv\\|.md` | statistiques descriptives et matrice "
        "de corrélation |\n",
        "Reproduction : `python -m src.download && python -m src.preprocessing && "
        "python -m scripts.export_assets`.\n",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parties))
    print(f"écrit : {OUT.relative_to(C.ROOT)}  ({len(OUT.read_text().split())} mots)")
    if synthetique:
        print("ATTENTION : chiffres issus du jeu de substitution, non publiables.")


if __name__ == "__main__":
    main()
