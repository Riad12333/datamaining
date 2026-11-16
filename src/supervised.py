"""Étapes 1.4 et 2 : préparation supervisée et modèles de classification.

Instance : variables explicatives sélectionnées, transformées et extraites + calendrier.
Cible    : « 2-meter temperature » du mois SUIVANT, discrétisée en 3 intervalles
           de même largeur (froid / tempere / chaud).
Split    : chronologique — les `TEST_YEARS` dernières années en test.

Note de méthode : les bornes des intervalles et le StandardScaler sont calculés
sur le train seul puis appliqués au test pour éviter toute fuite d'information.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C

# Variables de base excluant les variables redondantes/colinéaires
DROPPED = getattr(C, "DROPPED_FEATURES", ["slhf", "ro"])
BASE_PREDICTORS = [c for c in C.PREDICTORS.values() if c not in DROPPED]

# Nouvelles features extraites et transformées
EXTRA_FEATURES = ["wind_speed", "rnet", "tp_log", "e_log"]
CALENDAR_FEATURES = ["year", "month", "month_sin", "month_cos"]
DEFAULT_FEATURES = [c for c in BASE_PREDICTORS if c not in ["tp", "e"]] + EXTRA_FEATURES + CALENDAR_FEATURES
FEATURES = list(DEFAULT_FEATURES)


# --------------------------------------------------------------------------
# Préparation
# --------------------------------------------------------------------------
def sync_features(df: pd.DataFrame) -> list[str]:
    """Adapte FEATURES aux variables effectivement présentes dans le DataFrame."""
    global FEATURES
    available = [c for c in DEFAULT_FEATURES if c in df.columns]
    missing = set(DEFAULT_FEATURES) - set(available)
    if missing:
        print(f"[info supervisé] variables absentes ignorées : {sorted(missing)}", flush=True)
    FEATURES = available
    return FEATURES


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features extraites/transformées et décale t2m d'un mois."""
    out = df.sort_values(["latitude", "longitude", "time"]).copy()

    # Feature extraction physique (T7)
    if "u10" in out and "v10" in out:
        out["wind_speed"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)
    if "ssr" in out and "str" in out:
        out["rnet"] = out["ssr"] + out["str"]

    # Transformations log1p sans paramètre (T3)
    if "tp" in out:
        out["tp_log"] = np.log1p(out["tp"])
    if "e" in out:
        out["e_log"] = np.log1p(-out["e"])

    # Encodage cyclique trigonométrique du mois
    if "month" in out:
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    feats = sync_features(out)
    g = out.groupby(["latitude", "longitude"], sort=False)

    out["t2m_target"] = g["t2m"].shift(-C.HORIZON_MONTHS)
    gap = g["time"].shift(-C.HORIZON_MONTHS) - out["time"]
    out.loc[gap > pd.Timedelta(days=32 * C.HORIZON_MONTHS), "t2m_target"] = np.nan

    return out.dropna(subset=feats + ["t2m_target"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame, test_years: int = C.TEST_YEARS):
    cutoff = int(df["year"].max()) - test_years + 1
    train = df[df["year"] < cutoff].reset_index(drop=True)
    test = df[df["year"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def equal_width_edges(values: pd.Series, n_bins: int = C.N_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Bornes de n intervalles de MÊME LARGEUR (et non de même effectif)."""
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    edges_ext = edges.copy()
    edges_ext[0], edges_ext[-1] = -np.inf, np.inf    # le test peut sortir de l'étendue train
    return edges_ext, edges


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

    # StandardScaler ajusté STRICTEMENT sur le TRAIN seul
    scaler = StandardScaler()
    scaler.fit(train[FEATURES])

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, getattr(C, "FILE_SCALER", C.DATA_PROCESSED / "scaler.joblib"))
    train.to_parquet(getattr(C, "FILE_TRAIN_PARQUET", C.DATA_PROCESSED / "train.parquet"), index=False)
    test.to_parquet(getattr(C, "FILE_TEST_PARQUET", C.DATA_PROCESSED / "test.parquet"), index=False)

    meta = {
        "features": FEATURES,
        "dropped_features": DROPPED,
        "extracted_features": ["wind_speed", "rnet"],
        "cutoff_year_test": cutoff,
        "edges_C": [round(float(b), 2) for b in shown],
        "classes": C.BIN_LABELS,
        "n_train": len(train),
        "n_test": len(test)
    }
    feat_json_path = getattr(C, "FILE_FEATURES_JSON", C.DATA_PROCESSED / "features.json")
    feat_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.concat([train.assign(split="train"), test.assign(split="test")],
              ignore_index=True).to_parquet(C.FILE_SUPERVISED, index=False)

    if verbose:
        print(f"bornes équi-largeur (°C, calées sur le train) : {np.round(shown, 2)}", flush=True)
        print(f"split chronologique : train < {cutoff} ({len(train):,} lignes) | "
              f"test >= {cutoff} ({len(test):,} lignes)", flush=True)
        print("répartition des classes (train) :", flush=True)
        print(train["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS).round(3).to_string(), flush=True)
        print("répartition des classes (test) :", flush=True)
        print(test["classe"].value_counts(normalize=True).reindex(C.BIN_LABELS).round(3).to_string(), flush=True)
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

    def knn():
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
          f"sur {len(train):,} (coût du SVM ; comparaison à effectif égal)", flush=True)
    return train.sample(n=n, random_state=seed).reset_index(drop=True)


def evaluate(train: pd.DataFrame, test: pd.DataFrame,
             sample_size: int | None = C.SVM_MAX_TRAIN, models: dict | None = None):
    """Entraîne tous les modèles sur le même échantillon et renvoie le tableau comparatif."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, cohen_kappa_score,
                                 confusion_matrix, f1_score)

    sync_features(train)
    models = models or make_models()
    tr = subsample(train, sample_size)
    Xtr, ytr = tr[FEATURES], tr["classe"]
    Xte, yte = test[FEATURES], test["classe"]

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
        print(f"  {name:<14} acc={rows[-1]['accuracy']:.4f}  "
              f"f1_macro={rows[-1]['f1_macro']:.4f}  ({fit_s:.1f}s)", flush=True)

    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    return table, fitted, reports


if __name__ == "__main__":
    tr, te, _ = prepare()
    print("\nentraînement :", flush=True)
    # Pour un test rapide en ligne de commande, on évalue Random Forest et baseline
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    quick_models = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                n_jobs=-1, random_state=C.RANDOM_STATE)
    }
    table, fitted, reports = evaluate(tr, te, models=quick_models)
    print("\n=== comparaison ===", flush=True)
    print(table.round(4).to_string(index=False), flush=True)
    best = table.loc[0, "modele"]
    print(f"\n=== meilleur modèle : {best} ===", flush=True)
    print(reports[best]["report"], flush=True)
    print(pd.DataFrame(reports[best]["confusion"], index=C.BIN_LABELS, columns=C.BIN_LABELS), flush=True)
