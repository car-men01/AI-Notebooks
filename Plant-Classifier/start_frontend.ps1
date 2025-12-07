# Start Frontend Script
# This script starts the React frontend

Write-Host "Starting Plant Classifier Frontend..." -ForegroundColor Green

# Navigate to frontend directory
cd frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    npm install
}

# Start the development server
Write-Host "Starting React dev server on http://localhost:3000..." -ForegroundColor Green
npm start
