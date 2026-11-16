"""Inspecte les jeux de données parquet produits par la chaîne.

    python -m scripts.voir                      # liste tout
    python -m scripts.voir supervised           # schéma + aperçu + stats
    python -m scripts.voir clean -n 20          # 20 lignes
    python -m scripts.voir supervised --csv     # export CSV ouvrable dans Excel
    python -m scripts.voir supervised -q "year == 2025 and classe == 'froid'"

Les parquet font des centaines de milliers de lignes : l'export CSV est
volontairement un échantillon, pas le fichier entier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C   # noqa: E402

JEUX = {
    "clean": (C.FILE_CLEAN, "données nettoyées — une ligne par (point, mois)"),
    "supervised": (C.FILE_SUPERVISED, "étape 2 — instances supervisées, cible t+1"),
    "unsupervised": (C.FILE_UNSUPERVISED, "étape 3 — profils (point, année)"),
    "clusters": (C.FILE_CLUSTERS, "étape 3 — étiquettes de cluster par année"),
}
APERCU = C.ROOT / "reports" / "apercu"


def taille(path: Path) -> str:
    o = path.stat().st_size
    for u in ("o", "Ko", "Mo", "Go"):
        if o < 1024:
            return f"{o:.0f} {u}"
        o /= 1024
    return f"{o:.1f} To"


def lister() -> None:
    import pyarrow.parquet as pq

    lignes = []
    for nom, (path, desc) in JEUX.items():
        if not path.exists():
            lignes.append({"jeu": nom, "état": "absent", "lignes": "—",
                           "colonnes": "—", "taille": "—", "contenu": desc})
            continue
        # Lire les métadonnées suffit : inutile de charger le fichier en mémoire.
        md = pq.ParquetFile(path).metadata
        lignes.append({"jeu": nom, "état": "ok", "lignes": f"{md.num_rows:,}".replace(",", " "),
                       "colonnes": md.num_columns, "taille": taille(path), "contenu": desc})
    print(pd.DataFrame(lignes).to_string(index=False))
    print(f"\ndétail : python -m scripts.voir <jeu>")


def detailler(nom: str, n: int, requete: str | None, csv: bool, colonnes: str | None) -> None:
    path, desc = JEUX[nom]
    if not path.exists():
        raise SystemExit(f"{path} absent — lancer la chaîne d'abord "
                         f"(python -m src.preprocessing)")

    df = pd.read_parquet(path)
    if requete:
        avant = len(df)
        df = df.query(requete)
        print(f"filtre « {requete} » : {avant:,} -> {len(df):,} lignes\n".replace(",", " "))
        if df.empty:
            raise SystemExit("aucune ligne ne correspond")
    if colonnes:
        garde = [c.strip() for c in colonnes.split(",")]
        manquantes = [c for c in garde if c not in df.columns]
        if manquantes:
            raise SystemExit(f"colonnes inconnues : {manquantes}")
        df = df[garde]

    print(f"=== {nom} — {desc} ===")
    print(f"fichier : {path.relative_to(C.ROOT)}  ({taille(path)})")
    print(f"forme   : {len(df):,} lignes × {df.shape[1]} colonnes".replace(",", " "))
    print(f"mémoire : {df.memory_usage(deep=True).sum() / 1024**2:.0f} Mo une fois chargé")

    print("\n--- colonnes ---")
    schema = pd.DataFrame({
        "type": df.dtypes.astype(str),
        "non nuls": df.notna().sum(),
        "nuls": df.isna().sum(),
        "valeurs distinctes": [df[c].nunique() for c in df.columns],
    })
    print(schema.to_string())

    print(f"\n--- {min(n, len(df))} premières lignes ---")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.head(n).to_string())

    num = df.select_dtypes("number")
    if not num.empty:
        print("\n--- statistiques (colonnes numériques) ---")
        with pd.option_context("display.width", 250):
            print(num.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3).to_string())

    categorielles = [c for c in df.columns
                     if not pd.api.types.is_numeric_dtype(df[c])
                     and not pd.api.types.is_datetime64_any_dtype(df[c])
                     and df[c].nunique() <= 30]
    for c in categorielles:
        print(f"\n--- répartition de `{c}` ---")
        print(df[c].value_counts().to_string())

    if csv:
        APERCU.mkdir(parents=True, exist_ok=True)
        out = APERCU / f"{nom}_apercu.csv"
        ech = df if len(df) <= 5000 else df.sample(5000, random_state=C.RANDOM_STATE).sort_index()
        ech.to_csv(out, index=False)
        print(f"\nécrit : {out.relative_to(C.ROOT)}  ({len(ech):,} lignes"
              .replace(",", " ") +
              (" — échantillon aléatoire, le fichier complet est trop gros pour un tableur)"
               if len(ech) < len(df) else ")"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jeu", nargs="?", choices=list(JEUX), help="jeu à inspecter")
    p.add_argument("-n", "--rows", type=int, default=10, help="lignes à afficher (défaut 10)")
    p.add_argument("-q", "--query", help="filtre pandas, ex. \"year == 2025\"")
    p.add_argument("-c", "--cols", help="colonnes à garder, séparées par des virgules")
    p.add_argument("--csv", action="store_true", help="exporter un aperçu CSV")
    a = p.parse_args()

    lister() if a.jeu is None else detailler(a.jeu, a.rows, a.query, a.csv, a.cols)
