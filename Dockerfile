FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8888

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y

COPY src/ src/
COPY notebooks/ notebooks/
COPY README.md .
RUN mkdir -p data/raw data/processed reports/figures

# Jeu de substitution prétraité : l'image est utilisable sans clé CDS.
# Le NetCDF brut (~140 Mo) est retiré de l'image ; `python -m src.download
# --synthetic` le régénère à l'identique si besoin.
RUN python -m src.download --synthetic \
    && python -m src.preprocessing \
    && rm -f data/raw/*.nc

EXPOSE 8888

CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=${PORT:-8888} --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir=/app"]
