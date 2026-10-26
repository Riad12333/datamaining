"""Étape 4 : préparation supervisée.

Cible   : t2m décalée d'un mois (t+1), discrétisée en 3 intervalles de même largeur.
Split   : chronologique (les dernières années en test, jamais de mélange aléatoire).

Les bornes des intervalles sont calées sur le train seul, puis appliquées au test :
découper sur l'ensemble complet ferait fuir l'information du futur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FEATURES = [
    "t2m", "str", "t2m_lag1", "t2m_lag2", "str_lag1",
    "latitude", "longitude", "month_sin", "month_cos",
]


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute lags, encodage cyclique du mois et la cible continue t+1."""
    out = df.sort_values(["latitude", "longitude", "time"]).copy()
    g = out.groupby(["latitude", "longitude"], sort=False)

    out["t2m_lag1"] = g["t2m"].shift(1)
    out["t2m_lag2"] = g["t2m"].shift(2)
    out["str_lag1"] = g["str"].shift(1)
    out["t2m_target"] = g["t2m"].shift(-C.HORIZON_MONTHS)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out.dropna(subset=FEATURES + ["t2m_target"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_years: int = C.TEST_YEARS):
    cutoff = df["year"].max() - test_years + 1
    train = df[df["year"] < cutoff].reset_index(drop=True)
    test = df[df["year"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def equal_width_edges(values: pd.Series, n_bins: int = C.N_BINS) -> np.ndarray:
    """Bornes de n intervalles de MÊME LARGEUR (et non de même effectif)."""
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf   # le test peut sortir de l'étendue train
    return edges


def discretize(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(values, bins=edges, labels=C.BIN_LABELS, include_lowest=True)


def prepare(df: pd.DataFrame | None = None):
    if df is None:
        df = pd.read_parquet(C.FILE_CLEAN)

    frame = build_frame(df)
    train, test, cutoff = chronological_split(frame)

    edges = equal_width_edges(train["t2m_target"])
    train["classe"] = discretize(train["t2m_target"], edges)
    test["classe"] = discretize(test["t2m_target"], edges)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    full = pd.concat(
        [train.assign(split="train"), test.assign(split="test")], ignore_index=True
    )
    full.to_parquet(C.FILE_SUPERVISED, index=False)

    print(f"bornes (°C) : {np.round(np.linspace(train['t2m_target'].min(), train['t2m_target'].max(), C.N_BINS + 1), 2)}")
    print(f"split chronologique : train < {cutoff} ({len(train):,}) | test >= {cutoff} ({len(test):,})")
    print(train["classe"].value_counts(normalize=True).round(3).to_string())
    return train, test, edges


def evaluate(train: pd.DataFrame, test: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.tree import DecisionTreeClassifier

    Xtr, ytr = train[FEATURES], train["classe"]
    Xte, yte = test[FEATURES], test["classe"]

    models = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "arbre": DecisionTreeClassifier(max_depth=8, random_state=C.RANDOM_STATE),
        "foret": RandomForestClassifier(
            n_estimators=200, max_depth=14, n_jobs=-1, random_state=C.RANDOM_STATE
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        results[name] = model
        print(f"\n=== {name} ===")
        print(classification_report(yte, pred, zero_division=0))
        print(confusion_matrix(yte, pred, labels=C.BIN_LABELS))
    return results


if __name__ == "__main__":
    tr, te, _ = prepare()
    evaluate(tr, te)
