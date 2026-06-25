# ─────────────────────────────────────────────────────────────────
# Road-AI  –  Dockerfile
# ─────────────────────────────────────────────────────────────────
# Build:   docker build -t road-ai .
# Run:     docker run -p 8000:8000 road-ai
# Demo:    docker run -p 8000:8000 road-ai python launch.py --demo --no-browser
# Video:   docker run -p 8000:8000 -v /path/to/video:/data road-ai \
#              python main.py --video /data/road.mp4 --dashboard
# ─────────────────────────────────────────────────────────────────

# Use slim Python 3.11 with OpenCV system deps pre-available
FROM python:3.11-slim

LABEL maintainer="Road-AI"
LABEL description="Road damage detection & analysis system"

# ── System dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Work directory ────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────
COPY *.py          ./
COPY dashboard.html ./

# ── Create output directories ─────────────────────────────────────
RUN mkdir -p output output/uploads output/maps output/tickets

# ── Expose dashboard port ─────────────────────────────────────────
EXPOSE 8000

# ── Default command: start the dashboard server ───────────────────
# Override with: docker run road-ai python launch.py --demo --no-browser
CMD ["python", "dashboard_server.py", "--port", "8000"]