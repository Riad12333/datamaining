"""Étapes 1.5 et 3 : préparation non supervisée et clustering.

Instance : un couple (année, point de grille).
  - t2m_01..t2m_12  : les 12 valeurs mensuelles de température (°C)
  - str_01..str_12  : les 12 valeurs mensuelles de rayonnement thermique net (W m-2)
  - tendance intra-annuelle, par variable : _mean, _std, _amplitude, _pente
  - tendance long terme, par variable : _anomalie (écart à la normale du point,
    calculée sur les `BASELINE_YEARS` premières années) et _derive (pente des
    moyennes annuelles du point sur toute la période)

Algorithmes appliqués année par année : K-Means, DBSCAN, AGNES (agglomératif
Ward) et AutoEncoder + K-Means.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config as C

VARS = ["t2m", "str"]
KEYS = ["latitude", "longitude", "year"]
_MONTH_X = np.arange(1, 13, dtype=float)
_MONTH_XC = _MONTH_X - _MONTH_X.mean()
_MONTH_DEN = float((_MONTH_XC ** 2).sum())


def _ols_slope(block: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pente OLS ligne par ligne (vectorisée)."""
    xc = x - x.mean()
    yc = block - block.mean(axis=1, keepdims=True)
    return (yc * xc).sum(axis=1) / float((xc ** 2).sum())


# --------------------------------------------------------------------------
# Préparation des profils annuels
# --------------------------------------------------------------------------
def build_year_profiles(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_parquet(C.FILE_CLEAN)

    frames = []
    for var in VARS:
        wide = df.pivot_table(index=KEYS, columns="month", values=var)
        wide = wide.reindex(columns=range(1, 13)).dropna()      # années incomplètes écartées
        block = wide.to_numpy()
        wide.columns = [f"{var}_{m:02d}" for m in range(1, 13)]

        wide[f"{var}_mean"] = block.mean(axis=1)
        wide[f"{var}_std"] = block.std(axis=1, ddof=0)
        wide[f"{var}_amplitude"] = block.max(axis=1) - block.min(axis=1)
        wide[f"{var}_pente"] = _ols_slope(block, _MONTH_X)
        frames.append(wide)

    out = frames[0].join(frames[1], how="inner").reset_index()
    out = _add_longterm_features(out)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(C.FILE_UNSUPERVISED, index=False)
    print(f"écrit : {C.FILE_UNSUPERVISED}  ({len(out):,} profils (année × point), "
          f"{len(feature_columns(out))} features)")
    return out


def _add_longterm_features(df: pd.DataFrame, baseline_years: int = C.BASELINE_YEARS):
    """Anomalie par rapport à la normale du point, et dérive pluriannuelle."""
    out = df.sort_values(["latitude", "longitude", "year"]).copy()
    years = np.sort(out["year"].unique())
    base = years[:baseline_years]

    for var in VARS:
        col = f"{var}_mean"
        normale = (
            out[out["year"].isin(base)]
            .groupby(["latitude", "longitude"])[col].mean()
            .rename(f"{var}_normale")
        )
        out = out.merge(normale, on=["latitude", "longitude"], how="left")
        out[f"{var}_anomalie"] = out[col] - out[f"{var}_normale"]
        out = out.drop(columns=f"{var}_normale")

        derive = (
            out.groupby(["latitude", "longitude"])
            .apply(lambda g: _ols_slope(g[col].to_numpy()[None, :],
                                        g["year"].to_numpy(float))[0],
                   include_groups=False)
            .rename(f"{var}_derive")
        )
        out = out.merge(derive, on=["latitude", "longitude"], how="left")

    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.split("_")[0] in VARS]


# --------------------------------------------------------------------------
# AutoEncoder (sans dépendance deep learning : MLP sklearn à goulot)
# --------------------------------------------------------------------------
def _encode(ae, X: np.ndarray) -> np.ndarray:
    """Propagation jusqu'à la couche goulot d'un MLPRegressor entraîné en auto-encodeur."""
    h = X
    for i in range(2):                                  # entrée -> h1 -> goulot
        h = h @ ae.coefs_[i] + ae.intercepts_[i]
        h = np.maximum(h, 0) if ae.activation == "relu" else np.tanh(h)
    return h


def autoencoder_embedding(X: np.ndarray, bottleneck: int = C.AE_BOTTLENECK) -> np.ndarray:
    from sklearn.neural_network import MLPRegressor

    ae = MLPRegressor(
        hidden_layer_sizes=(32, bottleneck, 32),
        activation="relu", solver="adam", learning_rate_init=1e-3,
        max_iter=400, early_stopping=True, n_iter_no_change=15,
        random_state=C.RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ae.fit(X, X)                                    # reconstruction de l'entrée
    return _encode(ae, X)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------
def k_distances(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Distances triées au k-ième voisin — la courbe dont on cherche le coude."""
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def _dbscan_space(X: np.ndarray) -> np.ndarray:
    """DBSCAN raisonne sur des densités : on réduit d'abord la dimension (ACP 95 %)."""
    from sklearn.decomposition import PCA

    return PCA(n_components=0.95, random_state=C.RANDOM_STATE).fit_transform(X)


def fit_dbscan(Z: np.ndarray, min_samples: int = C.DBSCAN_MIN_SAMPLES,
               max_noise: float = C.DBSCAN_MAX_NOISE):
    """Choisit eps sur la courbe des k-distances puis renvoie (labels, eps).

    Le coude seul donne un eps trop petit dès que le nuage est un continuum :
    on balaie donc les centiles de la courbe et on retient le eps qui maximise
    la silhouette sous contrainte de bruit. Repli, dans l'ordre : le eps le
    moins bruité parmi ceux donnant au moins deux groupes, puis le eps médian
    — auquel cas DBSCAN ne voit qu'une seule classe de densité, ce qui est un
    résultat en soi et doit être rapporté tel quel.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    y = k_distances(Z, min_samples)
    runs = []
    for p in range(5, 100, 5):
        eps = float(np.percentile(y, p))
        if eps <= 0:
            continue
        lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        keep = lab >= 0
        runs.append({"eps": eps, "labels": lab,
                     "k": len(np.unique(lab[keep])), "bruit": float((~keep).mean())})

    multi = [r for r in runs if r["k"] >= 2]
    sous_seuil = [r for r in multi if r["bruit"] <= max_noise]
    if sous_seuil:
        best = max(sous_seuil,
                   key=lambda r: silhouette_score(Z[r["labels"] >= 0],
                                                  r["labels"][r["labels"] >= 0]))
    elif multi:
        best = min(multi, key=lambda r: r["bruit"])
    else:
        best = runs[len(runs) // 2]
    return best["labels"], best["eps"]


def _canonical(labels: np.ndarray, order_by: np.ndarray) -> np.ndarray:
    """Renumérote les clusters par `order_by` croissant (bruit DBSCAN laissé à -1).

    Sans cela les étiquettes sont arbitraires et incomparables d'une année à l'autre.
    """
    valid = labels >= 0
    uniq = np.unique(labels[valid])
    means = {c: order_by[valid & (labels == c)].mean() for c in uniq}
    rank = {c: i for i, c in enumerate(sorted(uniq, key=means.get))}
    return np.array([rank.get(l, -1) for l in labels])


def cluster_one_year(X: np.ndarray, order_by: np.ndarray, k: int = C.N_CLUSTERS) -> dict:
    from sklearn.cluster import AgglomerativeClustering, KMeans

    def dbscan():
        return fit_dbscan(_dbscan_space(X))[0]

    algos = {
        "kmeans": lambda: KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit_predict(X),
        "agnes": lambda: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
        "dbscan": dbscan,
        "ae_kmeans": lambda: KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE)
                             .fit_predict(autoencoder_embedding(X)),
    }
    return {name: _canonical(fn(), order_by) for name, fn in algos.items()}


def score(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)

    mask = labels >= 0
    n_clusters = len(np.unique(labels[mask]))
    out = {"n_clusters": n_clusters, "bruit": float((~mask).mean())}
    if n_clusters < 2 or mask.sum() < 3:
        return {**out, "silhouette": np.nan, "calinski": np.nan, "davies_bouldin": np.nan}
    return {
        **out,
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "calinski": calinski_harabasz_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
    }


def run_all_years(profiles: pd.DataFrame | None = None, k: int = C.N_CLUSTERS):
    """Clustering année par année. Renvoie (étiquettes, tableau de scores)."""
    from sklearn.preprocessing import StandardScaler

    if profiles is None:
        profiles = pd.read_parquet(C.FILE_UNSUPERVISED)

    cols = feature_columns(profiles)
    # Standardisation globale (et non par année) : les clusters restent
    # comparables d'une année à l'autre, ce qu'exige l'analyse d'évolution.
    scaled = pd.DataFrame(StandardScaler().fit_transform(profiles[cols]),
                          columns=cols, index=profiles.index)

    labels, scores = [], []
    for year, idx in profiles.groupby("year").groups.items():
        X = scaled.loc[idx].to_numpy()
        order_by = profiles.loc[idx, "t2m_mean"].to_numpy()
        res = cluster_one_year(X, order_by, k=k)

        frame = profiles.loc[idx, KEYS].copy()
        for algo, lab in res.items():
            frame[algo] = lab
            scores.append({"year": int(year), "algo": algo, **score(X, lab)})
        labels.append(frame)
        print(f"  {year} : " + "  ".join(
            f"{a}={len(np.unique(l[l >= 0]))}cl" for a, l in res.items()))

    labels = pd.concat(labels, ignore_index=True)
    labels.to_parquet(C.FILE_CLUSTERS, index=False)
    return labels, pd.DataFrame(scores)


# --------------------------------------------------------------------------
# Évolution sur la décennie
# --------------------------------------------------------------------------
def evolution(labels: pd.DataFrame, profiles: pd.DataFrame, algo: str = "kmeans"):
    """Effectif et température moyenne de chaque cluster, année par année."""
    merged = labels.merge(profiles[KEYS + ["t2m_mean", "t2m_anomalie", "str_mean"]], on=KEYS)
    tab = (
        merged[merged[algo] >= 0]
        .groupby(["year", algo])
        .agg(effectif=("t2m_mean", "size"),
             t2m_moyen=("t2m_mean", "mean"),
             anomalie=("t2m_anomalie", "mean"),
             str_moyen=("str_mean", "mean"))
        .reset_index()
        .rename(columns={algo: "cluster"})
    )
    tab["part"] = tab["effectif"] / tab.groupby("year")["effectif"].transform("sum")
    return tab


def transitions(labels: pd.DataFrame, algo: str = "kmeans") -> pd.DataFrame:
    """Matrice de transition des points entre la première et la dernière année."""
    y0, y1 = labels["year"].min(), labels["year"].max()
    a = labels[labels["year"] == y0].set_index(["latitude", "longitude"])[algo]
    b = labels[labels["year"] == y1].set_index(["latitude", "longitude"])[algo]
    joined = pd.concat([a.rename(str(y0)), b.rename(str(y1))], axis=1).dropna()
    return pd.crosstab(joined[str(y0)], joined[str(y1)], normalize="index").round(3)


def elbow(profiles: pd.DataFrame | None = None, ks=range(2, 9)) -> pd.DataFrame:
    """Coude et silhouette sur l'ensemble des profils, pour fixer k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if profiles is None:
        profiles = pd.read_parquet(C.FILE_UNSUPERVISED)
    X = StandardScaler().fit_transform(profiles[feature_columns(profiles)])

    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertie": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    profiles = build_year_profiles()
    print("\nclustering année par année :")
    labels, scores = run_all_years(profiles)

    print("\n=== scores moyens par algorithme ===")
    print(scores.groupby("algo")[["n_clusters", "bruit", "silhouette",
                                  "calinski", "davies_bouldin"]]
          .mean().round(3).to_string())

    print("\n=== évolution des clusters K-Means ===")
    print(evolution(labels, profiles).round(2).to_string(index=False))

    print(f"\n=== transitions {labels['year'].min()} -> {labels['year'].max()} (K-Means) ===")
    print(transitions(labels))
