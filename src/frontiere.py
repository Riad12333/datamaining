"""Découpage sur le contour administratif de l'Algérie.

L'emprise de téléchargement est un rectangle ; comme ERA5-Land ne masque que la
mer, elle conserve aussi le Maroc, la Tunisie, la Libye, le Mali, le Niger et la
Mauritanie. L'énoncé fixe la zone d'étude à l'Algérie : il faut donc un vrai
test point-dans-polygone, et non un découpage par bornes de latitude/longitude.

Contour : geoBoundaries gbOpen, niveau ADM0, dérivé d'OpenStreetMap
(ODbL 1.0). Récupéré par `telecharger_contour()`, mis en cache dans data/raw/.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

API = "https://www.geoboundaries.org/api/current/gbOpen/DZA/ADM0/"


def telecharger_contour(path=None, timeout: int = 90) -> "object":
    """Télécharge le contour si absent du cache, puis renvoie la géométrie."""
    from shapely.geometry import shape

    path = path or C.FILE_FRONTIERE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(API, timeout=timeout) as r:
            url = json.load(r)["gjDownloadURL"]
        print(f"[info] téléchargement du contour ADM0 de l'Algérie : {url}")
        urllib.request.urlretrieve(url, path)

    geo = json.loads(path.read_text())
    geoms = [shape(f["geometry"]) for f in geo["features"]]
    return geoms[0] if len(geoms) == 1 else __import__("shapely").union_all(geoms)


def masque_algerie(points: pd.DataFrame, buffer_deg: float = 0.0) -> np.ndarray:
    """Booléen par ligne : le centre de maille tombe-t-il en Algérie ?

    `buffer_deg` dilate le contour. À 0.4° de résolution une maille frontalière
    mesure ~44 km : son centre peut être hors frontière alors que la maille
    couvre en partie le territoire. Un buffer de l'ordre d'une demi-maille
    récupère ces mailles, au prix d'un peu de territoire voisin.
    """
    from shapely import points as shapely_points
    from shapely.prepared import prep

    contour = telecharger_contour()
    if buffer_deg:
        contour = contour.buffer(buffer_deg)

    # `prep` construit un index spatial : indispensable, le contour a des
    # milliers de sommets et un test naïf serait très lent.
    pret = prep(contour)
    pts = shapely_points(points["longitude"].to_numpy(),
                         points["latitude"].to_numpy())
    return np.fromiter((pret.contains(p) for p in pts), bool, len(pts))


def clip(df: pd.DataFrame, buffer_deg: float = C.FRONTIERE_BUFFER) -> pd.DataFrame:
    """Restreint le DataFrame long aux points de grille situés en Algérie."""
    pts = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    dedans = masque_algerie(pts, buffer_deg=buffer_deg)
    gardes = pts[dedans]

    out = df.merge(gardes, on=["latitude", "longitude"], how="inner")
    print(f"découpage sur le contour algérien : {len(pts):,} -> {len(gardes):,} "
          f"points de grille ({len(gardes)/len(pts):.0%} conservés, "
          f"buffer {buffer_deg}°)")
    return out.reset_index(drop=True)


def classer(points: pd.DataFrame, buffer_deg: float = C.FRONTIERE_BUFFER) -> pd.DataFrame:
    """Ajoute une colonne `zone` : algerie / frontalier / hors_algerie.

    Sert à la figure du masque : montrer ce qui est écarté est plus parlant que
    de ne montrer que ce qui est gardé.
    """
    strict = masque_algerie(points, buffer_deg=0.0)
    large = masque_algerie(points, buffer_deg=buffer_deg) if buffer_deg else strict
    out = points.copy()
    out["zone"] = np.select([strict, large & ~strict],
                            ["algerie", "frontalier"], default="hors_algerie")
    return out
