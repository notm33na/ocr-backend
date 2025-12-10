# Script to copy modules directory from UTRNet to ocr-backend
# Run this before building Docker image

$sourceDir = "..\UTRNet-High-Resolution-Urdu-Text-Recognition\modules"
$destDir = "modules"

if (Test-Path $sourceDir) {
    Write-Host "Copying modules directory from $sourceDir to $destDir..." -ForegroundColor Green
    
    if (Test-Path $destDir) {
        Write-Host "Removing existing modules directory..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $destDir
    }
    
    Copy-Item -Recurse $sourceDir $destDir
    Write-Host "✅ Modules directory copied successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Source directory not found: $sourceDir" -ForegroundColor Red
    Write-Host "Please ensure UTRNet-High-Resolution-Urdu-Text-Recognition directory exists." -ForegroundColor Yellow
}

