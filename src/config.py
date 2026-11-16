"""Paramètres partagés par les trois étapes du projet.

Sujet : prédiction de la classe de température à 2 m sur l'Algérie,
ERA5-Land moyennes mensuelles, 2015 → présent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
DIR_TABLES = ROOT / "reports" / "tables"

# --- Source ----------------------------------------------------------------
DATASET = "reanalysis-era5-land-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis"

# Emprise Algérie (Nord, Ouest, Sud, Est) avec une marge : la marge maritime
# produit les points NaN que l'étape de nettoyage doit écarter.
AREA = [37.5, -9.0, 18.5, 12.5]
NATIVE_GRID = 0.1          # résolution native ERA5-Land (degrés)
COARSEN_FACTOR = 2         # consigne prof : grouper 2x2 = 4 pixels ensemble (0.1° -> 0.2°)
MIN_PIXELS_BLOCK = 2       # consigne prof : garder un bloc avec >= 2 pixels algériens sur 4 (>= 50%)

# Features écartées pour colinéarité / quasi-constance (décision T3 / T5)
DROPPED_FEATURES = ["slhf", "ro"]
# Features extraites physiquement pertinentes (décision T7)
EXTRACTED_FEATURES = ["wind_speed", "rnet"]
# Période de bug ECMWF CDS sur les variables cumulées (décision T2)
ANOMALY_START = "2022-09-01"
ANOMALY_END = "2024-02-01"

YEAR_START = 2015
# Années complètes uniquement : le clustering travaille sur 12 mois par année.
YEAR_END = date.today().year - 1

# --- Variables -------------------------------------------------------------
# nom CDS -> nom court dans le NetCDF
TARGET_VAR = "t2m"

PREDICTORS = {
    "2m_dewpoint_temperature": "d2m",
    "surface_latent_heat_flux": "slhf",
    "surface_net_solar_radiation": "ssr",
    "surface_net_thermal_radiation": "str",
    "surface_sensible_heat_flux": "sshf",
    "runoff": "ro",
    "total_evaporation": "e",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
}
VARIABLES = {**PREDICTORS, "2m_temperature": TARGET_VAR}
SHORT_NAMES = list(VARIABLES.values())

# Unités : conversions appliquées à l'étape 1.
KELVIN_VARS = ["t2m", "d2m"]                       # K      -> °C
FLUX_VARS = ["slhf", "ssr", "str", "sshf"]         # J m-2  -> W m-2
DEPTH_VARS = ["ro", "e", "tp"]                     # m      -> mm/jour
PRESSURE_VARS = ["sp"]                             # Pa     -> hPa
SECONDS_PER_DAY = 86_400

# --- Étape 2 : supervisé ---------------------------------------------------
HORIZON_MONTHS = 1          # cible : t2m du mois suivant
N_BINS = 3                  # 3 intervalles de MÊME LARGEUR
BIN_LABELS = ["froid", "tempere", "chaud"]
TEST_YEARS = 2              # split chronologique : 2 dernières années en test
SVM_MAX_TRAIN = 30_000      # SVC est quadratique : sous-échantillon explicite

# --- Étape 3 : non supervisé ----------------------------------------------
N_CLUSTERS = 4
DBSCAN_MIN_SAMPLES = 10
DBSCAN_MAX_NOISE = 0.20   # part de bruit tolérée lors du choix de eps
AE_BOTTLENECK = 6
BASELINE_YEARS = 5          # 5 premières années = référence pour les anomalies

RANDOM_STATE = 42

# --- Fichiers --------------------------------------------------------------
FILE_RAW = DATA_RAW / "era5_land_algeria.nc"
FILE_DZ_JSON = ROOT / "data" / "dz.json"
FILE_CLEAN = DATA_PROCESSED / "era5_clean.parquet"
FILE_SUPERVISED = DATA_PROCESSED / "supervised.parquet"
FILE_TRAIN_PARQUET = DATA_PROCESSED / "train.parquet"
FILE_TEST_PARQUET = DATA_PROCESSED / "test.parquet"
FILE_SCALER = DATA_PROCESSED / "scaler.joblib"
FILE_FEATURES_JSON = DATA_PROCESSED / "features.json"
FILE_ANOMALIES_CSV = ROOT / "reports" / "tables" / "01_anomalies.csv"
FILE_UNSUPERVISED = DATA_PROCESSED / "unsupervised.parquet"
FILE_CLUSTERS = DATA_PROCESSED / "clusters_by_year.parquet"
