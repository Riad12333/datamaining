"""Étape 1-2 : téléchargement des données ERA5-Land mensuelles.

Usage :
    python -m src.download                 # via l'API Copernicus CDS (clé requise)
    python -m src.download --synthetic     # jeu de substitution hors-ligne

La clé CDS se place dans ~/.cdsapirc (ou CDSAPI_URL / CDSAPI_KEY).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

MONTHS = [f"{m:02d}" for m in range(1, 13)]


def download_cds() -> None:
    import cdsapi

    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(
        C.DATASET,
        {
            "product_type": C.PRODUCT_TYPE,
            "variable": list(C.VARIABLES),
            "year": [str(y) for y in range(C.YEAR_START, C.YEAR_END + 1)],
            "month": MONTHS,
            "time": "00:00",
            "area": C.AREA,
            "grid": C.GRID,
            "format": "netcdf",
        },
        str(C.FILE_RAW),
    )
    print(f"écrit : {C.FILE_RAW}")


def _land_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Masque terre/mer grossier : Méditerranée au nord, Atlantique à l'ouest."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    sea = (la > 37.0) | ((lo < -6.0) & (la > 28.0))
    return ~sea


def make_synthetic() -> None:
    """Reproduit la structure du jeu réel (mêmes colonnes, NaN en mer)."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    north, west, south, east = C.AREA
    lat = np.arange(south, north + 1e-9, 2.0)
    lon = np.arange(west, east + 1e-9, 2.0)
    land = _land_mask(lat, lon)

    dates = pd.date_range(f"{C.YEAR_START}-01-01", f"{C.YEAR_END}-12-01", freq="MS")
    rows = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if not land[i, j]:
                t2m = np.full(len(dates), np.nan)
                strad = np.full(len(dates), np.nan)
            else:
                base = 305.0 - 0.55 * (la - south)          # gradient nord-sud
                season = 11.0 * np.sin(2 * np.pi * (dates.month - 4) / 12)
                trend = 0.025 * (dates.year - C.YEAR_START)  # réchauffement
                t2m = base + season + trend + rng.normal(0, 1.4, len(dates))
                strad = -(5.4e6 + 7e4 * season + rng.normal(0, 2e5, len(dates)))
            rows.append(
                pd.DataFrame(
                    {"time": dates, "latitude": la, "longitude": lo,
                     "t2m": t2m, "str": strad}
                )
            )

    df = pd.concat(rows, ignore_index=True)
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = C.DATA_RAW / "era5_monthly.csv"
    df.to_csv(out, index=False)
    print(f"écrit : {out}  ({len(df):,} lignes, {df['t2m'].isna().mean():.1%} de NaN)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="génère un jeu de substitution sans appeler le CDS")
    a = p.parse_args()
    make_synthetic() if a.synthetic else download_cds()
