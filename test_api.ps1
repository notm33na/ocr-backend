# PowerShell test script for OCR Backend API
# Usage: .\test_api.ps1 <image_path> [language] [medical_corrections] [visualize]

param(
    [Parameter(Mandatory=$true)]
    [string]$ImagePath,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("english", "urdu")]
    [string]$Language = "english",
    
    [Parameter(Mandatory=$false)]
    [switch]$MedicalCorrections,
    
    [Parameter(Mandatory=$false)]
    [switch]$Visualize,
    
    [Parameter(Mandatory=$false)]
    [string]$ApiUrl = "http://localhost:8000"
)

# Check if image exists
if (-not (Test-Path $ImagePath)) {
    Write-Host "❌ Error: Image not found: $ImagePath" -ForegroundColor Red
    exit 1
}

Write-Host "📸 Loading image: $ImagePath" -ForegroundColor Cyan

# Encode image to base64
try {
    $imageBytes = [System.IO.File]::ReadAllBytes($ImagePath)
    $imageBase64 = [System.Convert]::ToBase64String($imageBytes)
    Write-Host "✅ Image encoded to base64 ($($imageBase64.Length) characters)" -ForegroundColor Green
} catch {
    Write-Host "❌ Error encoding image: $_" -ForegroundColor Red
    exit 1
}

# Prepare request payload
$payload = @{
    image = $imageBase64
    language = $Language
    medical_corrections = $MedicalCorrections.IsPresent
    visualize = $Visualize.IsPresent
} | ConvertTo-Json

$url = "$ApiUrl/ocr"

Write-Host ""
Write-Host "🚀 Sending request to: $url" -ForegroundColor Cyan
Write-Host "   Language: $Language"
Write-Host "   Medical corrections: $MedicalCorrections"
Write-Host "   Visualize: $Visualize"

try {
    # Send request
    $response = Invoke-RestMethod -Uri $url -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 300
    
    # Display result
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host "✅ OCR RESULT:" -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host $response.text
    Write-Host ("=" * 60) -ForegroundColor Green
    
    # Save visualization if provided
    if ($response.visualization) {
        $visPath = [System.IO.Path]::GetFileNameWithoutExtension($ImagePath) + "_visualization.png"
        $visBytes = [System.Convert]::FromBase64String($response.visualization)
        [System.IO.File]::WriteAllBytes($visPath, $visBytes)
        Write-Host ""
        Write-Host "💾 Visualization saved to: $visPath" -ForegroundColor Green
    }
    
    # Save text result
    $outputPath = [System.IO.Path]::GetFileNameWithoutExtension($ImagePath) + "_ocr_result.txt"
    $response.text | Out-File -FilePath $outputPath -Encoding UTF8
    Write-Host "💾 Text result saved to: $outputPath" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "❌ Error: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Red
    }
    exit 1
}

