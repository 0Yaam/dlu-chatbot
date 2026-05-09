FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

COPY app ./app
COPY .streamlit ./.streamlit
COPY data ./data
COPY dashboard.py ./
COPY .env.example ./.env.example

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data /app/vector_store /app/.cache/huggingface \
    && useradd --create-home --shell /bin/sh appuser \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["docker-entrypoint.sh"]


FROM base AS backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS dashboard

EXPOSE 8511

CMD ["python", "-m", "streamlit", "run", "dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8511", "--server.maxUploadSize", "512", "--server.maxMessageSize", "512"]
