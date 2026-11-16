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
    """Corrige le bug officiel ECMWF CDS (septembre 2022 à février 2024) sur les flux cumulés.

    Les variables cumulées (ssr, str, slhf, e, ro, tp) ont été tronquées à 12h au lieu de 24h.
    On applique le ratio climatologique spatial mensuel sur chaque mois affecté.
    """
    out = df.copy()
    vars_present = [v for v in C.SHORT_NAMES if v in out.columns]
    accum_vars = [v for v in ["ssr", "str", "slhf", "e", "ro", "tp"] if v in vars_present]

    out["_year"] = out["time"].dt.year
    out["_month"] = out["time"].dt.month

    # Climatologie normale (années train hors période anomalie)
    normal_mask = (out["_year"] < 2025) & ~((out["time"] >= C.ANOMALY_START) & (out["time"] <= C.ANOMALY_END))
    clim_mean = out[normal_mask].groupby("_month")[accum_vars].mean()
    clim_std = out[normal_mask].groupby("_month")[accum_vars].std()

    affected_mask = (out["time"] >= C.ANOMALY_START) & (out["time"] <= C.ANOMALY_END)
    affected_times = sorted(out.loc[affected_mask, "time"].unique())

    log_rows = []
    n_pts = out.groupby(["latitude", "longitude"]).ngroups

    for t in affected_times:
        t_mask = (out["time"] == t)
        m = pd.to_datetime(t).month
        for v in accum_vars:
            actual_mean = float(out.loc[t_mask, v].mean())
            norm_mean = float(clim_mean.loc[m, v])
            norm_std = float(clim_std.loc[m, v])
            z_score = (actual_mean - norm_mean) / norm_std if norm_std != 0 else 0
            ratio = actual_mean / norm_mean if norm_mean != 0 else 1.0

            factor = norm_mean / actual_mean if actual_mean != 0 else 1.0
            out.loc[t_mask, v] = out.loc[t_mask, v] * factor

            log_rows.append({
                "variable": v,
                "mois": pd.to_datetime(t).strftime("%Y-%m"),
                "valeur_brute": round(actual_mean, 4),
                "normale_mensuelle": round(norm_mean, 4),
                "z_score": round(z_score, 2),
                "ratio_val_sur_normale": round(ratio, 3),
                "regle_appliquee": f"Correction facteur echelle ECMWF (x{factor:.3f})",
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
