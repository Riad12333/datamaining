"""Étapes 1.4 et 2 : préparation supervisée et modèles de classification.

Instance : les 11 variables explicatives du sujet + mois + année.
Cible    : « 2-meter temperature » du mois SUIVANT, discrétisée en 3 intervalles
           de même largeur (froid / tempere / chaud).
Split    : chronologique — les `TEST_YEARS` dernières années en test.

Note de méthode : les bornes des intervalles sont calculées sur le train seul
puis appliquées au test. Les découper sur l'ensemble complet ferait fuir dans le
train une information sur l'étendue des températures futures.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C

BASE_FEATURES = list(C.PREDICTORS.values())            # les 11 variables du sujet
DERIVED_FEATURES = ["sshf_bilan"]      # substitut de sshf, cf. preprocessing
CALENDAR_FEATURES = ["year", "month", "month_sin", "month_cos"]
FEATURES = BASE_FEATURES + CALENDAR_FEATURES


def features_for(df: pd.DataFrame) -> list[str]:
    """Variables effectivement disponibles, dans l'ordre nominal du sujet.

    Un export CDS incomplet ne doit pas faire échouer la chaîne sur un KeyError
    opaque : on travaille avec ce qui est là, et on dit ce qui manque.
    """
    presentes = [v for v in BASE_FEATURES if v in df.columns]
    absentes = [v for v in BASE_FEATURES if v not in df.columns]
    derivees = [v for v in DERIVED_FEATURES if v in df.columns]
    if absentes:
        print(f"ATTENTION — variables explicatives absentes des données : {absentes}. "
              f"{len(presentes)}/{len(BASE_FEATURES)} variables du sujet utilisées"
              + (f", complétées par {derivees}." if derivees else "."))
    return presentes + derivees + CALENDAR_FEATURES


# --------------------------------------------------------------------------
# Préparation
# --------------------------------------------------------------------------
def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Décale t2m d'un mois à l'intérieur de chaque point de grille."""
    out = df.sort_values(["latitude", "longitude", "time"]).copy()
    g = out.groupby(["latitude", "longitude"], sort=False)

    out["t2m_target"] = g["t2m"].shift(-C.HORIZON_MONTHS)
    # Le décalage doit porter sur deux mois consécutifs : un trou dans la série
    # (point partiellement manquant) invaliderait la cible.
    gap = g["time"].shift(-C.HORIZON_MONTHS) - out["time"]
    out.loc[gap > pd.Timedelta(days=32 * C.HORIZON_MONTHS), "t2m_target"] = np.nan

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    feats = features_for(out)
    return out.dropna(subset=feats + ["t2m_target"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_years: int = C.TEST_YEARS):
    cutoff = int(df["year"].max()) - test_years + 1
    train = df[df["year"] < cutoff].reset_index(drop=True)
    test = df[df["year"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def equal_width_edges(values: pd.Series, n_bins: int = C.N_BINS) -> np.ndarray:
    """Bornes de n intervalles de MÊME LARGEUR (et non de même effectif)."""
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf    # le test peut sortir de l'étendue train
    return edges, np.linspace(lo, hi, n_bins + 1)


def discretize(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(values, bins=edges, labels=C.BIN_LABELS, include_lowest=True)


def prepare(df: pd.DataFrame | None = None, verbose: bool = True):
    if df is None:
        df = pd.read_parquet(C.FILE_CLEAN)

    frame = build_frame(df)
    train, test, cutoff = chronological_split(frame)

    edges, shown = equal_width_edges(train["t2m_target"])
    train["classe"] = discretize(train["t2m_target"], edges)
    test["classe"] = discretize(test["t2m_target"], edges)

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.concat([train.assign(split="train"), test.assign(split="test")],
              ignore_index=True).to_parquet(C.FILE_SUPERVISED, index=False)

    if verbose:
        print(f"bornes équi-largeur (°C, calées sur le train) : {np.round(shown, 2)}")
        print(f"split chronologique : train < {cutoff} ({len(train):,} lignes) | "
              f"test >= {cutoff} ({len(test):,} lignes)")
        print("répartition des classes (train) :")
        print(train["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS).round(3).to_string())
    return train, test, edges


# --------------------------------------------------------------------------
# Modèles
# --------------------------------------------------------------------------
def make_models(n_jobs: int = -1) -> dict:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    rs = C.RANDOM_STATE

    def knn():    # distances -> standardisation obligatoire
        return make_pipeline(StandardScaler(),
                             KNeighborsClassifier(n_neighbors=15, weights="distance",
                                                  n_jobs=n_jobs))

    def svm(proba=False):
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=10.0, gamma="scale",
                                 probability=proba, cache_size=1000, random_state=rs))

    def forest():
        return RandomForestClassifier(n_estimators=300, max_depth=None,
                                      min_samples_leaf=2, n_jobs=n_jobs, random_state=rs)

    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "knn": knn(),
        "svm": svm(),
        "random_forest": forest(),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=200, learning_rate=0.5, random_state=rs),
        "vote_hard": VotingClassifier(
            [("knn", knn()), ("svm", svm()), ("rf", forest())], voting="hard"),
        "vote_soft": VotingClassifier(
            [("knn", knn()), ("svm", svm(proba=True)), ("rf", forest())], voting="soft"),
    }


def subsample(train: pd.DataFrame, n: int | None, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Sous-échantillon aléatoire du train — SVC est quadratique en nombre de lignes."""
    if n is None or n >= len(train):
        return train
    print(f"[info] entraînement sur un sous-échantillon de {n:,} lignes "
          f"sur {len(train):,} (coût du SVM ; comparaison à effectif égal)")
    return train.sample(n=n, random_state=seed).reset_index(drop=True)


def evaluate(train: pd.DataFrame, test: pd.DataFrame,
             sample_size: int | None = C.SVM_MAX_TRAIN, models: dict | None = None):
    """Entraîne tous les modèles sur le même échantillon et renvoie le tableau comparatif."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, cohen_kappa_score,
                                 confusion_matrix, f1_score)

    models = models or make_models()
    feats = features_for(train)
    tr = subsample(train, sample_size)
    Xtr, ytr = tr[feats], tr["classe"]
    Xte, yte = test[feats], test["classe"]

    rows, fitted, reports = [], {}, {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        pred = model.predict(Xte)

        fitted[name] = model
        reports[name] = {
            "report": classification_report(yte, pred, labels=C.BIN_LABELS, zero_division=0),
            "confusion": confusion_matrix(yte, pred, labels=C.BIN_LABELS),
        }
        rows.append({
            "modele": name,
            "accuracy": accuracy_score(yte, pred),
            "balanced_acc": balanced_accuracy_score(yte, pred),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "f1_pondere": f1_score(yte, pred, average="weighted", zero_division=0),
            "kappa": cohen_kappa_score(yte, pred),
            "temps_fit_s": round(fit_s, 1),
        })
        print(f"  {name:<14} acc={rows[-1]['accuracy']:.3f}  "
              f"f1_macro={rows[-1]['f1_macro']:.3f}  ({fit_s:.1f}s)")

    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return table, fitted, reports


if __name__ == "__main__":
    tr, te, _ = prepare()
    print("\nentraînement :")
    table, fitted, reports = evaluate(tr, te)
    print("\n=== comparaison ===")
    print(table.round(3).to_string(index=False))
    best = table.loc[0, "modele"]
    print(f"\n=== meilleur modèle : {best} ===")
    print(reports[best]["report"])
    print(pd.DataFrame(reports[best]["confusion"], index=C.BIN_LABELS, columns=C.BIN_LABELS))
