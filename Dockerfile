FROM python:3.11-slim

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
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

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Railway will set PORT env var dynamically)
EXPOSE 8080

# Start the FastAPI server using startup script
CMD ["/bin/bash", "./start.sh"]

