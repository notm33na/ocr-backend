FROM python:3.11-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port (Railway will set PORT env var dynamically)
EXPOSE 8080

# Start the FastAPI server (use PORT env var, default to 8080)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

