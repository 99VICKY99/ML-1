@echo off
echo ========================================
echo    AQI Analysis Project Runner
echo ========================================
echo.

echo [1/3] Checking Python environment...
if not exist ".conda\python.exe" (
    echo ERROR: Conda environment not found!
    echo Please run setup_environment.bat first
    pause
    exit /b 1
)

echo [2/3] Running AQI Analysis...
echo This may take a few minutes...
echo.

".conda\python.exe" 1_optimized.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo    SUCCESS! Analysis Complete!
    echo ========================================
    echo.
    echo Generated files:
    echo - aqi_analysis.png
    echo - model_comparison.png  
    echo - shap_summary.png
    echo.
    echo [3/3] Opening results...
    start aqi_analysis.png
    start model_comparison.png
    start shap_summary.png
) else (
    echo.
    echo ========================================
    echo    ERROR! Analysis Failed!
    echo ========================================
    echo Please check the error messages above
)

echo.
pause
