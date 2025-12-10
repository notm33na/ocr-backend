#!/bin/bash
# Installation script for OCR Backend
# This script installs dependencies including CPU-only PyTorch

echo "Installing OCR Backend dependencies..."

# Install PyTorch CPU-only first
echo "Installing PyTorch (CPU-only)..."
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0

# Install remaining dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Installation complete!"

