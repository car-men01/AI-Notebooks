# Start Backend Script
# This script starts the FastAPI backend with the correct working directory

Write-Host "Starting Plant Classifier Backend..." -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Warning: .env file not found. Make sure environment variables are set." -ForegroundColor Yellow
}

# Check if virtual environment exists and is valid
$venvPython = ".venv\Scripts\python.exe"
if (Test-Path ".venv") {
    if (-not (Test-Path $venvPython)) {
        Write-Host "Virtual environment is corrupted. Removing and recreating..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force .venv
    }
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Start the server (run from PROJECT ROOT, not backend directory)
Write-Host "Starting FastAPI server on http://localhost:8080..." -ForegroundColor Green
Write-Host "API Documentation available at http://localhost:8080/docs" -ForegroundColor Cyan
uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload