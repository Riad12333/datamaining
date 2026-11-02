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

# --- Source ----------------------------------------------------------------
DATASET = "reanalysis-era5-land-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis"

# Emprise Algérie (Nord, Ouest, Sud, Est) avec une marge : la marge maritime
# produit les points NaN que l'étape de nettoyage doit écarter.
AREA = [37.5, -9.0, 18.5, 12.5]
NATIVE_GRID = 0.1          # résolution native ERA5-Land (degrés)
COARSEN_FACTOR = 4         # « reduce the spatial resolution by a factor of four »

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

# Substitut de `sshf` par le bilan énergétique de surface, quand la variable
# est absente de l'export CDS : ssr + str + slhf + sshf + G = 0, avec G (flux
# dans le sol) négligeable en moyenne mensuelle. Mettre à False pour s'en tenir
# strictement aux variables téléchargées.
ADD_ENERGY_BALANCE = True

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
FILE_CLEAN = DATA_PROCESSED / "era5_clean.parquet"
FILE_SUPERVISED = DATA_PROCESSED / "supervised.parquet"
FILE_UNSUPERVISED = DATA_PROCESSED / "unsupervised.parquet"
FILE_CLUSTERS = DATA_PROCESSED / "clusters_by_year.parquet"
