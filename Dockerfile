# Base image with Python 3.10 / PyTorch compatibility
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Copy requirement files first for optimal Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and outputs (model weights) into container
COPY src /app/src
COPY configs /app/configs
COPY outputs /app/outputs

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default environment variable for device
ENV DEVICE=cpu

# Default command to start FastAPI backend
CMD ["python", "-m", "uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]