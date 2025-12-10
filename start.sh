#!/bin/bash
# Startup script for OCR Backend
# Reads PORT from environment variable, defaults to 8080

PORT=${PORT:-8080}
echo "Starting OCR Backend API on port $PORT"

exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 1

