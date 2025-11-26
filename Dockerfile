# Stage 1: Base
FROM python:3.13-slim as base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app \
  HF_HOME=/app/.cache/huggingface \
  TORCH_HOME=/app/.cache/torch \
  SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Stage 2: Builder
FROM base as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install to system paths (since we copy /usr/local later)
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Test
FROM builder as test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY app/ app/
COPY tests/ tests/
COPY pytest.ini .
COPY scripts/docker-entrypoint-test.sh /entrypoint.sh

RUN mkdir -p /app/data/test /app/.cache/test && \
    chmod +x /entrypoint.sh && \
    chmod -R 777 /app/data /app/.cache

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "pytest"]

# Stage 4: Release
FROM base as release

COPY --from=builder /usr/local /usr/local

COPY app/ app/
COPY scripts/docker-entrypoint.sh /entrypoint.sh

RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

RUN mkdir -p /app/data /app/.cache /home/appuser/.streamlit && \
    chown -R appuser:appuser /app /home/appuser && \
    chmod +x /entrypoint.sh

ENV HOME=/home/appuser

USER appuser

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "app.main"]
