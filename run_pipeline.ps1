# SYNTHLA-EDU Pipeline Runner (PowerShell)
# Run this script in PowerShell with: .\run_pipeline.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SYNTHLA-EDU Pipeline Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "Installing/updating dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed"
    }
    Write-Host "Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create output directories
Write-Host ""
Write-Host "Creating output directories..." -ForegroundColor Yellow
$directories = @("clean", "synthetic", "results")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "Exists: $dir" -ForegroundColor Gray
    }
}

# Check for OULAD data
Write-Host ""
Write-Host "Checking for OULAD data..." -ForegroundColor Yellow
if (!(Test-Path "OULAD data")) {
    Write-Host "WARNING: 'OULAD data' directory not found" -ForegroundColor Yellow
    Write-Host "Please ensure your OULAD CSV files are in the 'OULAD data' directory" -ForegroundColor Yellow
    Read-Host "Press Enter to continue anyway"
} else {
    $csvFiles = Get-ChildItem "OULAD data" -Filter "*.csv"
    Write-Host "Found $($csvFiles.Count) CSV files in OULAD data directory" -ForegroundColor Green
}

# Run the pipeline
Write-Host ""
Write-Host "Starting SYNTHLA-EDU pipeline..." -ForegroundColor Yellow
Write-Host "This may take 1-2 hours depending on your system..." -ForegroundColor Yellow
Write-Host ""
Write-Host "As of v1.0, ML utility evaluation is handled in main, while quality and privacy are handled in the evaluation suite for efficiency." -ForegroundColor Yellow

$startTime = Get-Date
try {
    python oulad_synthetic_analysis.py
    if ($LASTEXITCODE -ne 0) {
        throw "Pipeline execution failed"
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: Pipeline failed to complete" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Pipeline completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Yellow
Write-Host "- clean/oulad_master_engineered.csv" -ForegroundColor White
Write-Host "- synthetic/synthetic_*.csv" -ForegroundColor White
Write-Host "- results/final_results.json" -ForegroundColor White
Write-Host "- results/summary_*.png" -ForegroundColor White
Write-Host "- results/quality_report_*.png" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit" 
