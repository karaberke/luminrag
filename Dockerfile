FROM python:3.11-slim@sha256:ae952e0a57138a4317b9b61caaa19a8217c2f356803f8d7e63aad08769950937

WORKDIR /app

# Self-hosted image includes the optional audio/video processors.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gosu \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock requirements-bootstrap.lock ./
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements-bootstrap.lock \
    && pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements.lock
ENV ORT_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/models \
    HF_HOME=/var/cache/luminrag/huggingface \
    XDG_CACHE_HOME=/var/cache/luminrag \
    LUMIN_MODEL_THREADS=2 \
    HOME=/home/lumin

COPY backend/utils/models.py backend/utils/model_manifest.json backend/utils/
COPY scripts/cache_models.py scripts/cache_models.py
RUN python scripts/cache_models.py

COPY . .

RUN useradd --create-home --uid 1000 lumin \
    && mkdir -p backend/data/processed backend/data/raw backend/data/staging /var/cache/luminrag \
    && chown -R root:root /app /opt/models /home/lumin \
    && chmod -R a-w /app /opt/models /home/lumin \
    && chmod -R a+rX /app /opt/models /home/lumin \
    && chown lumin:lumin backend/data/processed backend/data/raw backend/data/staging /var/cache/luminrag \
    && chmod 700 backend/data/processed backend/data/raw backend/data/staging /var/cache/luminrag \
    && chmod 555 scripts/docker-entrypoint.sh
ENTRYPOINT ["scripts/docker-entrypoint.sh"]

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000
CMD ["python", "-m", "backend"]
