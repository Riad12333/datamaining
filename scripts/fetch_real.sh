#!/usr/bin/env bash
# Télécharge le vrai jeu ERA5-Land depuis le Copernicus CDS, puis relance tout
# le prétraitement. Survit à une déconnexion SSH (setsid + nohup).
#
#   ./scripts/fetch_real.sh          # lance en tâche de fond
#   tail -f logs/fetch_real.log      # suivre
#
# Prérequis : ~/.cdsapirc (voir README).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/logs/fetch_real.log"
RAW="$ROOT/data/raw/era5_land_algeria.nc"

mkdir -p "$ROOT/logs" "$ROOT/data/raw"

[ -x "$PY" ] || { echo "venv absent : python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
"$PY" -c "import sys; sys.path.insert(0, '$ROOT'); from src.download import check_credentials; check_credentials()"

# Le fichier courant est le jeu de substitution : on l'écarte sans l'écraser.
if [ -f "$RAW" ] && "$PY" - <<EOF
import sys, xarray as xr
sys.exit(0 if "substitution" in xr.open_dataset("$RAW").attrs.get("source", "") else 1)
EOF
then
    mv "$RAW" "$ROOT/data/raw/synthetique_$(date +%Y%m%d%H%M).nc"
    echo "jeu de substitution mis de côté"
fi

run() {
    cd "$ROOT"
    echo "=== $(date -Is) téléchargement CDS ==="
    "$PY" -m src.download
    echo "=== $(date -Is) prétraitement ==="
    "$PY" -m src.preprocessing
    echo "=== $(date -Is) supervisé ==="
    "$PY" -m src.supervised
    echo "=== $(date -Is) clustering ==="
    "$PY" -m src.clustering
    echo "=== $(date -Is) terminé ==="
}

export -f run 2>/dev/null || true
setsid nohup bash -c "$(declare -f run); run" > "$LOG" 2>&1 < /dev/null &
echo "lancé (pid $!) — suivi : tail -f $LOG"
