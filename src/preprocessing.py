"""Étape 1.2 à 1.3 : réduction spatiale, exploration, nettoyage.

- `coarsen` : agrégation par blocs 4×4 (moyenne) -> résolution 0.1° -> 0.4°
- `to_long` : passage du cube NetCDF à un DataFrame (time, lat, lon, 12 variables)
- `drop_sea_points` : les points entièrement NaN sont hors ERA5-Land (mer, hors zone)
- `convert_units` : K -> °C, J m-2 -> W m-2, m -> mm/jour, Pa -> hPa
- helpers d'EDA : taux de NaN, statistiques descriptives, corrélations
"""

from __future__ import annotations

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

    # `number` et `expver` arrivent en coordonnées scalaires ou par pas de temps
    # selon l'export CDS ; elles pollueraient le DataFrame long.
    out = out.drop_vars([c for c in ("number", "expver") if c in out.coords])
    return out


def coarsen(ds: xr.Dataset, factor: int = C.COARSEN_FACTOR) -> xr.Dataset:
    """Réduit la résolution spatiale d'un facteur `factor` par moyenne de blocs.

    `boundary="trim"` écarte les lignes/colonnes en trop plutôt que de produire
    un bloc partiel. La moyenne de xarray ignore les NaN : un bloc à cheval sur
    la côte garde la moyenne de ses points terrestres, un bloc entièrement en
    mer reste NaN et sera écarté par `drop_sea_points`.
    """
    if factor <= 1:
        return ds
    out = ds.coarsen(latitude=factor, longitude=factor, boundary="trim").mean()
    before, after = ds["latitude"].size * ds["longitude"].size, out["latitude"].size * out["longitude"].size
    print(f"réduction spatiale {factor}×{factor} : {before:,} -> {after:,} points de grille")
    return out


def to_long(ds: xr.Dataset) -> pd.DataFrame:
    df = ds.to_dataframe().reset_index()
    cols = ["time", "latitude", "longitude"] + [v for v in C.SHORT_NAMES if v in df]
    df = df[cols].copy()
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["latitude", "longitude", "time"], ignore_index=True)


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


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Taux de NaN de t2m par point de grille et délimitation frontière Algérie."""
    rep = (
        df.groupby(["latitude", "longitude"])["t2m"]
        .apply(lambda s: s.isna().mean())
        .rename("nan_ratio")
        .reset_index()
    )
    rep["type"] = np.select(
        [rep["nan_ratio"] > 0.99, rep["nan_ratio"] > 0],
        ["hors_terre", "partiel"],
        default="terre",
    )
    paths = load_algeria_paths()
    if paths:
        pts = rep[["longitude", "latitude"]].to_numpy()
        inside = np.zeros(len(pts), dtype=bool)
        for p in paths:
            inside |= p.contains_points(pts)
        rep.loc[~inside & (rep["type"] != "hors_terre"), "type"] = "hors_algerie"
    return rep


def drop_sea_points(df: pd.DataFrame) -> pd.DataFrame:
    """Écarte les points hors terre et hors Algérie, interpole les trous isolés par point."""
    rep = missing_report(df)
    land = rep.loc[rep["type"].isin(["terre", "partiel"]), ["latitude", "longitude"]]
    out = df.merge(land, on=["latitude", "longitude"], how="inner")
    value_cols = [c for c in C.SHORT_NAMES if c in out]
    out[value_cols] = (
        out.groupby(["latitude", "longitude"], sort=False)[value_cols]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    return out.dropna(subset=value_cols).reset_index(drop=True)


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


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    return out


def complete_years(df: pd.DataFrame, min_months: int = 12) -> pd.DataFrame:
    """Restreint aux années disposant des 12 mois.

    Une année partielle biaise toute moyenne annuelle : si seuls janvier à
    septembre sont présents, la moyenne est tirée vers le haut par l'absence
    des mois froids, ce qui fausse la tendance pluriannuelle.
    """
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


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in C.SHORT_NAMES if c in df]
    return df[cols].corr(numeric_only=True)


def build(path=None) -> pd.DataFrame:
    ds = coarsen(open_raw(path))
    df = add_calendar(convert_units(drop_sea_points(to_long(ds))))
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.FILE_CLEAN, index=False)
    n_pts = df.groupby(["latitude", "longitude"]).ngroups
    print(f"écrit : {C.FILE_CLEAN}  ({len(df):,} lignes, {n_pts:,} points de grille, "
          f"{df['year'].min()}–{df['year'].max()})")
    return df


if __name__ == "__main__":
    d = build()
    print(describe(d).round(2))
    print(correlations(d).round(2))
