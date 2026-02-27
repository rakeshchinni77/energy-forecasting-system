# Base Image (Lightweight Python with system libs)
FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for:
# pandas, numpy, prophet, matplotlib, torch
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project into container
COPY . .

# Create required directories (if not mounted)
RUN mkdir -p data/raw data/processed results logs

# Default command → run full ML pipeline
# Preprocess → Feature → Train → Evaluate
CMD python -m src.train --config config/main.yml && \
    python -m src.evaluate --config config/main.yml