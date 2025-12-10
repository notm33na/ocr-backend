#!/usr/bin/env python3
"""
Startup script for OCR Backend
Reads PORT from environment variable and starts uvicorn
"""
import os
import sys

# Get PORT from environment, default to 8080
port = int(os.getenv("PORT", "8080"))

print(f"Starting OCR Backend API on port {port}")

# Start uvicorn
import uvicorn
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=port,
    workers=1
)

