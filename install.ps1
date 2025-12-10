# Installation script for OCR Backend (Windows PowerShell)
# This script installs dependencies including CPU-only PyTorch

Write-Host "Installing OCR Backend dependencies..." -ForegroundColor Green

# Install PyTorch CPU-only first
Write-Host "Installing PyTorch (CPU-only)..." -ForegroundColor Yellow
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0

# Install remaining dependencies
Write-Host "Installing other dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "Installation complete!" -ForegroundColor Green

