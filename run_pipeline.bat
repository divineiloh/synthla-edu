@echo off
echo ========================================
echo SYNTHLA-EDU Pipeline Runner
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Creating output directories...
if not exist "clean" mkdir clean
if not exist "synthetic" mkdir synthetic
if not exist "results" mkdir results

echo.
echo Starting SYNTHLA-EDU pipeline...
echo This may take 1-2 hours depending on your system...
echo.
echo As of v1.0, ML utility evaluation is handled in main, while quality and privacy are handled in the evaluation suite for efficiency.
python oulad_synthetic_analysis.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Pipeline failed to complete
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Results saved to:
echo - clean/oulad_master_engineered.csv
echo - synthetic/synthetic_*.csv
echo - results/final_results.json
echo - results/summary_*.png
echo - results/quality_report_*.png
echo.
pause 
