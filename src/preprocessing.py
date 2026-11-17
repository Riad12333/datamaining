"""Étape 1.2 à 1.3 : réduction spatiale, exploration, nettoyage.

- `filter_algeria_blocks` : masque Algérie à 0.1° AVANT réduction 2×2 (Méthode B, min_pixels=2)
- `coarsen` : agrégation par blocs 2×2 (moyenne) -> résolution 0.1° -> 0.2°
- `to_long` : passage du cube NetCDF à un DataFrame (time, lat, lon, variables)
- `convert_units` : K -> °C, J m-2 -> W m-2, m -> mm/jour, Pa -> hPa
- `correct_ecmwf_anomalies` : correction d'échelle du bug ECMWF CDS (sept 2022 à fév 2024)
- helpers d'EDA : taux de NaN, statistiques descriptives, corrélations de Spearman
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from . import config as C


def open_raw(path=None) -> xr.Dataset:
    ds = xr.open_dataset(path or C.FILE_RAW)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:                      # ERA5T empilé avec les données finales
        ds = ds.mean("expver", skipna=True)

    present = [v for v in C.SHORT_NAMES if v in ds.data_vars]
    missing = [v for v in C.SHORT_NAMES if v not in present]
    if missing:
        print(f"ATTENTION — variables absentes du NetCDF : {missing}. "
              f"Le reste de la chaîne travaillera sans elles.")
    out = ds[present]

    out = out.drop_vars([c for c in ("number", "expver") if c in out.coords])
    return out


def load_algeria_paths(path=None) -> list:
    """Charge les polygones des frontières algériennes depuis dz.json."""
    import json
    import matplotlib.path as mpath

    p = path or getattr(C, "FILE_DZ_JSON", C.ROOT / "data" / "dz.json")
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        gj = json.load(f)
    paths = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords:
                paths.append(mpath.Path(ring))
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    paths.append(mpath.Path(ring))
    return paths


def filter_algeria_blocks(ds_raw: xr.Dataset, min_pixels: int = C.MIN_PIXELS_BLOCK,
                          factor: int = C.COARSEN_FACTOR) -> tuple[xr.Dataset, xr.DataArray]:
    """Méthode B : Masque Algérie à 0.1° AVANT réduction 2×2.

    Chaque bloc 2×2 ne fait la moyenne que sur ses pixels terrestres algériens.
    Un bloc est conservé s'il compte au moins `min_pixels` pixels algériens (>= 50% du bloc).
    """
    paths = load_algeria_paths()
    lats_raw = ds_raw.latitude.values
    lons_raw = ds_raw.longitude.values
    glon, glat = np.meshgrid(lons_raw, lats_raw)
    pts = np.column_stack([glon.ravel(), glat.ravel()])

    inside_raw = np.zeros(len(pts), dtype=bool)
    if paths:
        for p in paths:
            inside_raw |= p.contains_points(pts)
    inside_2d = inside_raw.reshape(len(lats_raw), len(lons_raw))

    da_inside = xr.DataArray(inside_2d, coords=[ds_raw.latitude, ds_raw.longitude],
                             dims=["latitude", "longitude"])
    count_dz = da_inside.astype(int).coarsen(latitude=factor, longitude=factor, boundary="trim").sum()

    # Masquage avant moyenne : les mailles hors Algérie deviennent NaN
    ds_masked = ds_raw.where(da_inside)
    ds_coarse = ds_masked.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()

    # Filtrage par nombre de pixels minimum
    ds_filtered = ds_coarse.where(count_dz >= min_pixels)
    n_kept = int((count_dz.values >= min_pixels).sum())
    print(f"Méthode B (masque 0.1° avant coarsen {factor}×{factor}, min_pixels={min_pixels}) : "
          f"{n_kept:,} points de grille retenus.")
    return ds_filtered, count_dz


def coarsen(ds: xr.Dataset, factor: int = C.COARSEN_FACTOR) -> xr.Dataset:
    """Réduit la résolution spatiale par moyenne de blocs (compatibilité)."""
    if factor <= 1:
        return ds
    out = ds.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    before, after = ds["latitude"].size * ds["longitude"].size, out["latitude"].size * out["longitude"].size
    print(f"réduction spatiale {factor}×{factor} : {before:,} -> {after:,} points de grille")
    return out


def to_long(ds: xr.Dataset) -> pd.DataFrame:
    df = ds.to_dataframe().reset_index()
    available_vars = [v for v in C.SHORT_NAMES if v in df.columns]
    cols = ["time", "latitude", "longitude"] + available_vars
    df = df[cols].dropna(subset=available_vars).copy()
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["latitude", "longitude", "time"], ignore_index=True)


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Conversions idempotentes : relancer le script ne convertit pas deux fois."""
    out = df.copy()
    for v in C.KELVIN_VARS:
        if v in out and out[v].median() > 150:
            out[v] = out[v] - 273.15
    for v in C.FLUX_VARS:
        if v in out and out[v].abs().median() > 1e4:
            out[v] = out[v] / C.SECONDS_PER_DAY
    for v in C.DEPTH_VARS:
        if v in out and out[v].abs().median() < 1.0:
            out[v] = out[v] * 1000.0
    for v in C.PRESSURE_VARS:
        if v in out and out[v].median() > 2000:
            out[v] = out[v] / 100.0
    return out


def correct_ecmwf_anomalies(df: pd.DataFrame, save_table_path: Path | None = C.FILE_ANOMALIES_CSV) -> pd.DataFrame:
    """Corrige le problème connu d'ECMWF (variables cumulées, 2022-09 à 2024-02, non corrigé dans le produit standard).

    La cause exacte n'étant pas documentée officiellement, on applique la méthode de secours :
    un facteur d'échelle CONSTANT par variable (inverse de la médiane des 18 ratios mensuels,
    calculé sur les années saines du train < 2025 hors période d'anomalie).
    Ce facteur est appliqué uniformément à tous les points et tous les mois de la fenêtre,
    ce qui évite la circularité (moyenne = normale) et préserve la dispersion et les anomalies réelles.
    """
    out = df.copy()
    vars_present = [v for v in C.SHORT_NAMES if v in out.columns]
    accum_vars = [v for v in ["ssr", "str", "slhf", "e", "ro", "tp"] if v in vars_present]

    out["_year"] = out["time"].dt.year
    out["_month"] = out["time"].dt.month

    # Climatologie normale sur le Train sain (< 2025 hors période anomalie)
    normal_mask = (out["_year"] < 2025) & ~((out["time"] >= C.ANOMALY_START) & (out["time"] <= C.ANOMALY_END))
    clim_mean = out[normal_mask].groupby("_month")[accum_vars].mean()
    clim_std = out[normal_mask].groupby("_month")[accum_vars].std()

    affected_mask = (out["time"] >= C.ANOMALY_START) & (out["time"] <= C.ANOMALY_END)
    affected_times = sorted(out.loc[affected_mask, "time"].unique())

    # Étape 1 : Calcul des ratios observés / normale pour chaque mois et variable
    monthly_data = []
    ratios_by_var = {v: [] for v in accum_vars}
    for t in affected_times:
        t_mask = (out["time"] == t)
        m = pd.to_datetime(t).month
        for v in accum_vars:
            act_mean = float(out.loc[t_mask, v].mean())
            norm_mean = float(clim_mean.loc[m, v])
            s = float(clim_std.loc[m, v])
            z = (act_mean - norm_mean) / s if s != 0 else 0.0
            r = act_mean / norm_mean if norm_mean != 0 else 1.0
            ratios_by_var[v].append(r)
            monthly_data.append({
                "time": t, "month": m, "var": v,
                "act_mean": act_mean, "norm_mean": norm_mean,
                "z_score": z, "ratio": r
            })

    # Étape 2 : Calcul du facteur constant = 1.0 / médiane(ratios) par variable
    const_factors = {}
    for v in accum_vars:
        med_r = float(np.median(ratios_by_var[v]))
        const_factors[v] = 1.0 / med_r if med_r != 0 else 1.0

    # Étape 3 : Application du facteur constant global sur la fenêtre
    n_pts = out.groupby(["latitude", "longitude"]).ngroups
    for v in accum_vars:
        out.loc[affected_mask, v] = out.loc[affected_mask, v] * const_factors[v]

    # Étape 4 : Journalisation des 108 mois-variables avec ratios avant et après correction
    log_rows = []
    for item in monthly_data:
        v = item["var"]
        t = item["time"]
        f_const = const_factors[v]
        val_corr = item["act_mean"] * f_const
        ratio_corr = val_corr / item["norm_mean"] if item["norm_mean"] != 0 else 1.0
        log_rows.append({
            "variable": v,
            "mois": pd.to_datetime(t).strftime("%Y-%m"),
            "valeur_brute": round(item["act_mean"], 4),
            "normale_mensuelle": round(item["norm_mean"], 4),
            "z_score": round(item["z_score"], 2),
            "ratio_val_sur_normale": round(item["ratio"], 4),
            "facteur_constant_applique": round(f_const, 4),
            "valeur_corrigee": round(val_corr, 4),
            "ratio_corrige_sur_normale": round(ratio_corr, 4),
            "commentaire": "problème connu d'ECMWF (variables cumulées, 2022-09 à 2024-02, non corrigé dans le produit standard)",
            "n_valeurs_modifiees": n_pts
        })

    out = out.drop(columns=["_year", "_month"])

    if save_table_path and log_rows:
        save_table_path.parent.mkdir(parents=True, exist_ok=True)
        df_log = pd.DataFrame(log_rows)
        df_log.to_csv(save_table_path, index=False, encoding="utf-8")
        save_table_path.with_suffix(".md").write_text(df_log.to_markdown(index=False), encoding="utf-8")
        print(f"[info anomalies] {len(df_log)} corrections appliquées sauvegardées dans {save_table_path}")

    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def complete_years(df: pd.DataFrame, min_months: int = 12) -> pd.DataFrame:
    n = df.groupby("year")["month"].nunique()
    gardees = n[n >= min_months].index
    ecartees = sorted(set(n.index) - set(gardees))
    if ecartees:
        print(f"[info] années incomplètes écartées du calcul de tendance : "
              f"{ecartees} ({', '.join(f'{y}: {n[y]} mois' for y in ecartees)})")
    return df[df["year"].isin(gardees)]


def describe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in C.SHORT_NAMES if c in df]
    return df[cols].describe().T


def correlations(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Matrice des corrélations.

    Par défaut, Spearman (demandé par la prof pour capturer les relations monotones non-linéaires).
    """
    cols = [c for c in C.SHORT_NAMES if c in df]
    return df[cols].corr(method=method, numeric_only=True)


def build(path=None) -> pd.DataFrame:
    """Exécute l'intégralité du prétraitement et produit era5_clean.parquet."""
    ds_raw = open_raw(path)
    ds_filtered, _ = filter_algeria_blocks(ds_raw, min_pixels=C.MIN_PIXELS_BLOCK)
    df_long = to_long(ds_filtered)
    df_units = convert_units(df_long)
    df_corr = correct_ecmwf_anomalies(df_units)
    df_final = add_calendar(df_corr)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(C.FILE_CLEAN, index=False)
    n_pts = df_final.groupby(["latitude", "longitude"]).ngroups
    print(f"écrit : {C.FILE_CLEAN}  ({len(df_final):,} lignes, {n_pts:,} points de grille, "
          f"{df_final['year'].min()}–{df_final['year'].max()})")
    print(f"Vérification NaN résiduels : {df_final.isna().sum().to_dict()}")
    return df_final


if __name__ == "__main__":
    d = build()
    print(describe(d).round(2))
    print(correlations(d).round(2))
