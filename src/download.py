"""Étape 1.1 : téléchargement ERA5-Land mensuel (Algérie, 2015 → dernière année complète).

    python -m src.download                 # API Copernicus CDS (clé dans ~/.cdsapirc)
    python -m src.download --synthetic     # jeu de substitution, même structure

Le NetCDF est écrit à la résolution native 0.1° ; la réduction 4×4 est faite à
l'étape suivante (src.preprocessing).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config as C

MONTHS = [f"{m:02d}" for m in range(1, 13)]


def download_cds(path=None) -> None:
    import cdsapi

    path = path or C.FILE_RAW
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve(
        C.DATASET,
        {
            "product_type": C.PRODUCT_TYPE,
            "variable": list(C.VARIABLES),
            "year": [str(y) for y in range(C.YEAR_START, C.YEAR_END + 1)],
            "month": MONTHS,
            "time": "00:00",
            "area": C.AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(path),
    )
    print(f"écrit : {path}")


def _algeria_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Contour grossier de l'Algérie : sert uniquement au jeu de substitution.

    Les données réelles portent leur propre masque (NaN en mer et hors ERA5-Land).
    """
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    med = la > 36.9 - 0.12 * np.clip(lo, -2, 9)      # côte méditerranéenne
    atl = (lo < -8.2) & (la > 27.0)                   # Atlantique / Maroc
    west = lo < -8.7 + 0.25 * np.clip(37 - la, 0, 12)
    east = lo > 9.0 + 0.30 * np.clip(37 - la, 0, 14)
    south = la < 19.0 + 0.25 * np.clip(lo, 0, 8)
    return ~(med | atl | west | east | south)


def make_synthetic(path=None, grid: float = C.NATIVE_GRID) -> None:
    """Grille 0.1°, 12 variables, NaN hors terres — structure identique au CDS."""
    path = path or C.FILE_RAW
    rng = np.random.default_rng(C.RANDOM_STATE)

    north, west, south, east = C.AREA
    lat = np.round(np.arange(north, south - 1e-9, -grid), 2)
    lon = np.round(np.arange(west, east + 1e-9, grid), 2)
    time = pd.date_range(f"{C.YEAR_START}-01-01", f"{C.YEAR_END}-12-01", freq="MS")

    land = _algeria_mask(lat, lon)[None, :, :]
    shape = (len(time), len(lat), len(lon))

    month = time.month.to_numpy()[:, None, None]
    year = time.year.to_numpy()[:, None, None]
    saison = np.sin(2 * np.pi * (month - 4) / 12)
    la = lat[None, :, None]

    def noisy(scale):
        return rng.normal(0, scale, shape).astype("float32")

    # Température : gradient nord-sud, saisonnalité, réchauffement ~0.04 °C/an.
    t2m = 301.0 - 0.75 * (la - south) + 11.5 * saison + 0.04 * (year - C.YEAR_START)
    t2m = t2m + noisy(1.3)
    # Champ d'humidité partiellement indépendant : sans lui toutes les variables
    # seraient de simples fonctions de la saison et la prédiction serait triviale.
    humidite = np.clip(0.55 - 0.02 * (la - south) + 0.10 * saison + noisy(0.14), 0.02, 0.95)

    fields = {
        "t2m": t2m,
        "d2m": t2m - 26.0 * (1 - humidite) + noisy(2.6),
        "slhf": -(1.1e6 * humidite + 3e5 * saison) + noisy(3.5e5),
        "ssr": 1.3e7 + 4.5e6 * saison - 3e4 * (la - south) + noisy(2.2e6),
        "str": -(5.4e6 + 7e5 * saison + 1.4e6 * humidite) + noisy(9e5),
        "sshf": -(2.4e6 + 9e5 * saison) + noisy(1.3e6),
        "ro": np.clip(2e-4 * humidite - 5e-5 * saison + noisy(1.4e-4), 0, None),
        "e": -(3e-3 * humidite + 1e-3 * saison) + noisy(1.1e-3),
        "u10": 1.4 + 1.1 * saison + noisy(2.6),
        "v10": -0.6 + 0.8 * saison + noisy(2.4),
        "sp": 101_325 - 11.0 * (la - south) ** 2 + noisy(650),
        "tp": np.clip(2.5e-3 * humidite - 1.2e-3 * saison + noisy(1.6e-3), 0, None),
    }

    ds = xr.Dataset(
        {
            name: (("valid_time", "latitude", "longitude"),
                   np.where(land, values, np.nan).astype("float32"))
            for name, values in fields.items()
        },
        coords={"valid_time": time, "latitude": lat, "longitude": lon},
    )
    ds.attrs["source"] = "jeu de substitution (src.download --synthetic)"

    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)

    nan = float(np.isnan(ds["t2m"].values).mean())
    print(f"écrit : {path}  grille {len(lat)}×{len(lon)}, "
          f"{len(time)} mois, {len(ds.data_vars)} variables, {nan:.0%} de NaN")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true",
                   help="génère un jeu de substitution sans appeler le CDS")
    a = p.parse_args()
    make_synthetic() if a.synthetic else download_cds()
