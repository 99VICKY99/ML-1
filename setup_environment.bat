@echo off
echo ========================================
echo    AQI Analysis Project Setup
echo ========================================
echo.

echo [1/4] Creating conda environment...
if exist ".conda" (
    echo Conda environment already exists, skipping...
) else (
    conda create --prefix .conda python=3.9 -y
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create conda environment
        pause
        exit /b 1
    )
)

echo [2/4] Installing required packages...
".conda\Scripts\pip.exe" install pandas numpy matplotlib seaborn scikit-learn xgboost shap

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)

echo [3/4] Checking if dataset exists...
if not exist "city_day.csv" (
    echo.
    echo WARNING: city_day.csv not found!
    echo Please download the Air Quality Data from:
    echo https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
    echo.
    echo For now, creating sample data for testing...
    ".conda\python.exe" -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
dates = pd.date_range(start='2015-01-01', end='2020-12-31', freq='D')
cities = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore'] * (len(dates) // 5 + 1)
cities = cities[:len(dates)]

data = {
    'Date': dates,
    'City': cities,
    'PM2.5': np.random.normal(50, 20, len(dates)).clip(0, 200),
    'PM10': np.random.normal(80, 30, len(dates)).clip(0, 300),
    'NO': np.random.normal(20, 10, len(dates)).clip(0, 100),
    'NO2': np.random.normal(30, 15, len(dates)).clip(0, 150),
    'NOx': np.random.normal(40, 20, len(dates)).clip(0, 200),
    'NH3': np.random.normal(25, 12, len(dates)).clip(0, 100),
    'CO': np.random.normal(1.5, 0.8, len(dates)).clip(0, 10),
    'SO2': np.random.normal(15, 8, len(dates)).clip(0, 80),
    'O3': np.random.normal(45, 25, len(dates)).clip(0, 200),
    'Benzene': np.random.normal(2, 1, len(dates)).clip(0, 10),
    'Toluene': np.random.normal(3, 1.5, len(dates)).clip(0, 15),
    'Xylene': np.random.normal(1.5, 0.8, len(dates)).clip(0, 8),
    'AQI': np.random.normal(100, 40, len(dates)).clip(0, 500),
    'AQI_Bucket': ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
}

# Create AQI_Bucket based on AQI values
aqi_buckets = []
for aqi in data['AQI']:
    if aqi <= 50:
        aqi_buckets.append('Good')
    elif aqi <= 100:
        aqi_buckets.append('Satisfactory')
    elif aqi <= 200:
        aqi_buckets.append('Moderate')
    elif aqi <= 300:
        aqi_buckets.append('Poor')
    elif aqi <= 400:
        aqi_buckets.append('Very Poor')
    else:
        aqi_buckets.append('Severe')

data['AQI_Bucket'] = aqi_buckets

df = pd.DataFrame(data)
df.to_csv('city_day.csv', index=False)
print('Sample dataset created successfully!')
"
)

echo [4/4] Setup complete!
echo.
echo ========================================
echo    Setup Complete!
echo ========================================
echo.
echo To run the analysis, use: run_project.bat
echo.
pause
