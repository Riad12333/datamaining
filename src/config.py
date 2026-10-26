"""Paramètres partagés par les trois étapes du projet."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# --- Source ERA5 -----------------------------------------------------------
# Jeu « ERA5-Land monthly averaged data » (Copernicus CDS) : les points en mer
# y sont masqués (NaN), ce qui est exactement le cas traité à l'étape 3.
DATASET = "reanalysis-era5-land-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis"

VARIABLES = {
    "2m_temperature": "t2m",                 # K      -> converti en °C
    "surface_net_thermal_radiation": "str",  # J m-2  -> converti en W m-2
}

# Emprise par défaut : Algérie + marge côtière/maritime (Nord, Ouest, Sud, Est).
# La marge en mer est volontaire : elle produit les NaN à nettoyer.
AREA = [38.5, -9.0, 18.0, 12.0]
GRID = [0.5, 0.5]

YEAR_START = 1991
YEAR_END = 2024

# --- Préparation supervisée ------------------------------------------------
HORIZON_MONTHS = 1       # on prédit t2m à t+1
N_BINS = 3               # discrétisation en 3 intervalles de même largeur
BIN_LABELS = ["froid", "tempere", "chaud"]
TEST_YEARS = 5           # les 5 dernières années servent de test (split chronologique)

# --- Préparation non supervisée -------------------------------------------
RANDOM_STATE = 42
N_CLUSTERS = 4

FILE_RAW = DATA_RAW / "era5_monthly.nc"
FILE_CLEAN = DATA_PROCESSED / "era5_clean.parquet"
FILE_SUPERVISED = DATA_PROCESSED / "supervised.parquet"
FILE_UNSUPERVISED = DATA_PROCESSED / "unsupervised.parquet"
