"""Étape 5 : préparation non supervisée.

Une ligne = (année, point de grille). Colonnes :
  - t2m_01..t2m_12  : les 12 valeurs mensuelles de température (°C)
  - str_01..str_12  : les 12 valeurs mensuelles de rayonnement thermique net (W m-2)
  - features de tendance par variable : moyenne, écart (std et amplitude), pente
    (coefficient de la régression linéaire sur les 12 mois, par mois)
Les couples (année, point) incomplets sont écartés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

VARS = ["t2m", "str"]
_X = np.arange(1, 13, dtype=float)
_XC = _X - _X.mean()
_DEN = float((_XC ** 2).sum())


def _slope(block: np.ndarray) -> np.ndarray:
    """Pente OLS mois par mois, vectorisée sur les lignes."""
    yc = block - block.mean(axis=1, keepdims=True)
    return (yc * _XC).sum(axis=1) / _DEN


def build_year_profiles(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(C.FILE_CLEAN)

    frames = []
    for var in VARS:
        wide = df.pivot_table(
            index=["latitude", "longitude", "year"], columns="month", values=var
        )
        wide = wide.reindex(columns=range(1, 13)).dropna()
        block = wide.to_numpy()
        wide.columns = [f"{var}_{m:02d}" for m in range(1, 13)]
        wide[f"{var}_mean"] = block.mean(axis=1)
        wide[f"{var}_std"] = block.std(axis=1, ddof=0)
        wide[f"{var}_amplitude"] = block.max(axis=1) - block.min(axis=1)
        wide[f"{var}_slope"] = _slope(block)
        frames.append(wide)

    out = frames[0].join(frames[1], how="inner").reset_index()
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.FILE_UNSUPERVISED, index=False)
    print(f"écrit : {C.FILE_UNSUPERVISED}  ({len(out):,} profils, {out.shape[1]} colonnes)")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.split("_")[0] in VARS]


def cluster(df: pd.DataFrame | None = None, k: int = C.N_CLUSTERS):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if df is None:
        df = pd.read_parquet(C.FILE_UNSUPERVISED)

    cols = feature_columns(df)
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95, random_state=C.RANDOM_STATE),
        KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE),
    )
    labels = pipe.fit_predict(df[cols])
    reduced = pipe[:-1].transform(df[cols])
    print(f"k={k}  silhouette={silhouette_score(reduced, labels):.3f}")

    out = df.copy()
    out["cluster"] = labels
    return out, pipe


def elbow(df: pd.DataFrame | None = None, ks=range(2, 9)) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if df is None:
        df = pd.read_parquet(C.FILE_UNSUPERVISED)
    X = StandardScaler().fit_transform(df[feature_columns(df)])

    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertie": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    profiles = build_year_profiles()
    labelled, _ = cluster(profiles)
    print(labelled.groupby("cluster")[["t2m_mean", "t2m_amplitude", "t2m_slope",
                                       "str_mean"]].mean().round(2))
