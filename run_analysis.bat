@echo off
echo ========================================
echo      Air Quality Analysis Script
echo ========================================
echo Starting analysis...
echo.

REM Change to the project directory
cd /d "c:\Users\vicky\Desktop\ML\proj-1"

REM Run the optimized Python script using conda
"c:\Users\vicky\Desktop\ML\proj-1\.conda\python.exe" 1_optimized.py

echo.
echo ========================================
echo Analysis Complete!
echo Check the generated PNG files:
echo - aqi_analysis.png
echo - model_comparison.png  
echo - shap_summary.png
echo ========================================
pause
