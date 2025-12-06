# =========================
# Stage 1: Build / Install deps
# =========================
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for psycopg2, librosa, soundfile, opencv, mediapipe, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt ./

# Install Python deps to a local user directory
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# =========================
# Stage 2: Runtime image
# =========================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# Runtime system libs (no build-essential needed here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy backend source
COPY backend /app

# Optional: create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck (FastAPI /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=3)" || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
