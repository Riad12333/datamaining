"""Étape 3 : exploration et nettoyage.

- lecture du NetCDF (ou du CSV de substitution) vers un DataFrame long
- repérage et suppression des points en mer (NaN sur toute la série)
- conversion des unités : K -> °C, J m-2 -> W m-2
- résumés : distributions, corrélations
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SECONDS_PER_DAY = 86_400


def load_raw() -> pd.DataFrame:
    """DataFrame long : time, latitude, longitude, t2m, str."""
    csv = C.DATA_RAW / "era5_monthly.csv"
    if C.FILE_RAW.exists():
        import xarray as xr

        ds = xr.open_dataset(C.FILE_RAW)
        if "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        df = ds[["t2m", "str"]].to_dataframe().reset_index()
    elif csv.exists():
        df = pd.read_csv(csv, parse_dates=["time"])
    else:
        raise FileNotFoundError(
            "aucune donnée brute — lancer `python -m src.download [--synthetic]`"
        )

    keep = ["time", "latitude", "longitude", "t2m", "str"]
    df = df[keep].copy()
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["latitude", "longitude", "time"], ignore_index=True)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Taux de NaN par point de grille — les points en mer sont à 100 %."""
    g = df.groupby(["latitude", "longitude"])
    rep = g["t2m"].apply(lambda s: s.isna().mean()).rename("nan_ratio").reset_index()
    rep["type"] = np.where(rep["nan_ratio"] > 0.99, "mer",
                           np.where(rep["nan_ratio"] > 0, "partiel", "terre"))
    return rep


def drop_sea_points(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les points entièrement manquants, interpole les trous isolés."""
    rep = missing_report(df)
    land = rep.loc[rep["type"] != "mer", ["latitude", "longitude"]]
    out = df.merge(land, on=["latitude", "longitude"], how="inner")
    out[["t2m", "str"]] = (
        out.groupby(["latitude", "longitude"])[["t2m", "str"]]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    return out.dropna(subset=["t2m", "str"]).reset_index(drop=True)


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """t2m : kelvin -> °C. str : J m-2 (cumul journalier) -> W m-2."""
    out = df.copy()
    if out["t2m"].median() > 150:          # idempotent : déjà en °C si < 150
        out["t2m"] = out["t2m"] - 273.15
    if out["str"].abs().median() > 1e4:
        out["str"] = out["str"] / SECONDS_PER_DAY
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["time"].dt.year
    out["month"] = out["time"].dt.month
    return out


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return df[["t2m", "str"]].describe().T


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["t2m", "str", "latitude", "longitude", "month"]
    return df[cols].corr(numeric_only=True)


def build() -> pd.DataFrame:
    df = add_calendar(convert_units(drop_sea_points(load_raw())))
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.FILE_CLEAN, index=False)
    print(f"écrit : {C.FILE_CLEAN}  ({len(df):,} lignes)")
    return df


if __name__ == "__main__":
    d = build()
    print(describe(d).round(2))
    print(correlations(d).round(3))
