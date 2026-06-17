# --- Build stage -------------------------------------------------------------
# A build-fázisban telepítjük a Python függőségeket egy izolált virtualenv-be.
# A torch CPU-only változatát használjuk: a Docker konténer Linux-on fut,
# ahol nincs sem MPS (csak macOS), sem CUDA (alapértelmezett base image).
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-időhöz szükséges rendszer-csomagok (pl. egyes Python wheel-ek build-jéhez)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# A teljes virtualenv ide kerül; a final image csak ezt másolja át.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# A requirements.txt-ből torch CPU-only változatot installálunk a hivatalos
# CPU index-ről, így spórolunk ~1.5 GB-ot a CUDA-bináris helyett.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install -r requirements.txt

# A modell letöltése a build során — így a konténer futtatáskor azonnal kész.
# A HF_HOME a non-root user home-jával fog egyezni a final image-ben.
ENV HF_HOME=/opt/hf_cache
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('facebook/bart-large-mnli'); \
    AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')"


# --- Final stage -------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/appuser/.cache/huggingface

# Non-root user — security best practice
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# A virtualenv és a HF modell-cache átmásolása a build-stage-ből
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/hf_cache /home/appuser/.cache/huggingface

WORKDIR /app

# Csak a futtatáshoz szükséges fájlok — a tests/, scripts/, notebook/ kimarad
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/prompts/ ./data/prompts/

USER appuser

EXPOSE 8000

# Healthcheck: a Docker daemon maga próbálkozik /health-en, és az image
# egészséges-egészségtelen állapotát jelzi. compose / k8s ezt használja.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
