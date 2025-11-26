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
# Note: .env.test must be mounted as a volume or provided as env vars at runtime

# Default command runs all tests including integration tests
# To skip integration tests: docker-compose ... python -m pytest -m "not integration"
CMD ["python", "-m", "pytest"]

# Stage 4: Release
FROM base as release

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY app/ app/

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create data and cache directories with correct permissions
RUN mkdir -p /app/data /app/.cache && chown -R appuser:appuser /app/data /app/.cache

USER appuser

EXPOSE 8501

ENTRYPOINT ["python", "-m"]
CMD ["app.main"]
