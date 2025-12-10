#!/usr/bin/env python3
"""
Startup script for OCR Backend
Reads PORT from environment variable and starts uvicorn
"""
import os
import sys

# Debug: Print all environment variables related to PORT
print("Environment variables:")
for key, value in os.environ.items():
    if "PORT" in key.upper():
        print(f"  {key}={value}")

# Get PORT from environment, default to 8080
port_str = os.getenv("PORT", "8080")
print(f"PORT from environment: '{port_str}'")
port = int(port_str)

print(f"Starting OCR Backend API on port {port}")

# Start uvicorn
import uvicorn
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=port,
    workers=1
)

