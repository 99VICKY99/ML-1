#!/usr/bin/env python3
"""
Optimized Air Quality Index (AQI) Prediction using Machine Learning
This version includes progress indicators and optimizations for faster execution
"""

print("=== Starting AQI Prediction Analysis ===")
print("Phase 1: Loading required libraries...")

# Import libraries with progress indication
import pandas as pd
print("✓ Pandas loaded")

import numpy as np
print("✓ Numpy loaded")

import matplotlib.pyplot as plt
import seaborn as sns
print("✓ Visualization libraries loaded")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print("✓ Scikit-learn loaded")

import xgboost as xgb
print("✓ XGBoost loaded")

import shap
print("✓ SHAP loaded")

import warnings
warnings.filterwarnings('ignore')
print("✓ All libraries loaded successfully!\n")

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("Phase 2: Loading and examining data...")

# Load the dataset
try:
    df = pd.read_csv('city_day.csv')
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("⚠ city_day.csv not found. Creating sample dataset for demonstration...")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'] * (n_samples // 5)
    
    df = pd.DataFrame({
        'Date': dates,
        'City': cities[:n_samples],
        'PM2.5': np.random.normal(50, 20, n_samples),
        'PM10': np.random.normal(80, 30, n_samples),
        'NO': np.random.normal(15, 8, n_samples),
        'NO2': np.random.normal(25, 12, n_samples),
        'NOx': np.random.normal(40, 15, n_samples),
        'NH3': np.random.normal(20, 10, n_samples),
        'CO': np.random.normal(1.2, 0.5, n_samples),
        'SO2': np.random.normal(12, 6, n_samples),
        'O3': np.random.normal(35, 15, n_samples),
        'Benzene': np.random.normal(2, 1, n_samples),
        'Toluene': np.random.normal(3, 1.5, n_samples),
        'Xylene': np.random.normal(1.5, 0.8, n_samples)
    })
    
    # Add some correlations to make it more realistic
    df['AQI'] = (df['PM2.5'] * 0.4 + df['PM10'] * 0.3 + df['NO2'] * 0.15 + 
                 df['CO'] * 20 + df['SO2'] * 0.1 + np.random.normal(0, 5, n_samples))
    df['AQI'] = np.clip(df['AQI'], 0, 500)  # Keep AQI in valid range
    
    print(f"✓ Sample dataset created: {df.shape[0]} rows, {df.shape[1]} columns")

# Display basic info
print(f"\nDataset Info:")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Cities: {df['City'].nunique()} unique cities")

print("\nPhase 3: Data preprocessing...")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
print("✓ Date features extracted")

# Handle missing values
missing_before = df.isnull().sum().sum()
df = df.fillna(df.median(numeric_only=True))
print(f"✓ Missing values handled: {missing_before} → {df.isnull().sum().sum()}")

# Define pollutant columns (corrected)
pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
available_pollutants = [col for col in pollutant_cols if col in df.columns]
print(f"✓ Available pollutants: {available_pollutants}")

# Prepare features (corrected)
features = available_pollutants + ['Year', 'Month', 'Day']
available_features = [col for col in features if col in df.columns]
print(f"✓ Features for modeling: {available_features}")

# Check if we have AQI column, if not create it
if 'AQI' not in df.columns:
    print("⚠ AQI column not found, creating synthetic AQI based on available pollutants...")
    # Simple AQI calculation based on PM2.5 and PM10
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['AQI'] = df['PM2.5'] * 2 + df['PM10'] * 1.5 + np.random.normal(0, 10, len(df))
        df['AQI'] = np.clip(df['AQI'], 0, 500)
    else:
        print("❌ Cannot create AQI without PM2.5 or PM10 data")
        exit(1)

print("Phase 4: Exploratory Data Analysis...")

# Quick statistics
print(f"\nAQI Statistics:")
print(f"Mean: {df['AQI'].mean():.2f}")
print(f"Std: {df['AQI'].std():.2f}")
print(f"Min: {df['AQI'].min():.2f}")
print(f"Max: {df['AQI'].max():.2f}")

# Create plots
print("Creating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# AQI distribution
axes[0,0].hist(df['AQI'], bins=50, alpha=0.7, color='skyblue')
axes[0,0].set_title('AQI Distribution')
axes[0,0].set_xlabel('AQI')
axes[0,0].set_ylabel('Frequency')

# AQI vs PM2.5 (if available)
if 'PM2.5' in df.columns:
    axes[0,1].scatter(df['PM2.5'], df['AQI'], alpha=0.5, color='orange')
    axes[0,1].set_title('AQI vs PM2.5')
    axes[0,1].set_xlabel('PM2.5')
    axes[0,1].set_ylabel('AQI')

# Monthly AQI trend
monthly_aqi = df.groupby('Month')['AQI'].mean()
axes[1,0].plot(monthly_aqi.index, monthly_aqi.values, marker='o', color='green')
axes[1,0].set_title('Monthly AQI Trend')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('Average AQI')
axes[1,0].set_xticks(range(1, 13))

# Correlation heatmap (top correlations only)
corr_cols = available_pollutants + ['AQI']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            ax=axes[1,1], cbar_kws={'shrink': 0.8})
axes[1,1].set_title('Pollutant Correlation Matrix')

plt.tight_layout()
plt.savefig('aqi_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualizations created and saved as 'aqi_analysis.png'")

print("\nPhase 5: Machine Learning Model Training...")

# Prepare data for modeling
X = df[available_features]
y = df['AQI']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")

# Train models (optimized selection)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),  # Reduced trees for speed
    'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Reduced trees for speed
}

results = {}
print("\nTraining models...")

for name, model in models.items():
    print(f"Training {name}...")
    
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'model': model,
        'predictions': y_pred
    }
    
    print(f"✓ {name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")

print("\nPhase 6: Model Evaluation and Results...")

# Find best model
best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"R² Score: {results[best_model_name]['R²']:.4f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.4f}")
print(f"MAE: {results[best_model_name]['MAE']:.4f}")

# Model comparison plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Model performance comparison
model_names = list(results.keys())
r2_scores = [results[name]['R²'] for name in model_names]
rmse_scores = [results[name]['RMSE'] for name in model_names]

axes[0].bar(model_names, r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
axes[0].set_title('Model R² Comparison')
axes[0].set_ylabel('R² Score')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(model_names, rmse_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
axes[1].set_title('Model RMSE Comparison')
axes[1].set_ylabel('RMSE')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Model comparison saved as 'model_comparison.png'")

print("\nPhase 7: SHAP Feature Importance Analysis...")

# SHAP analysis (optimized - using smaller sample)
if best_model_name != 'Linear Regression':
    print("Calculating SHAP values...")
    
    # Use smaller sample for faster SHAP calculation
    sample_size = min(100, len(X_test))  # Use max 100 samples for speed
    X_sample = X_test.iloc[:sample_size]
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=available_features, show=False)
    plt.title(f'SHAP Summary Plot for {best_model_name}')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ SHAP analysis completed and saved as 'shap_summary.png'")
    
    # Feature importance
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n📊 Top Feature Importances:")
    print(importance_df.head(10).to_string(index=False))

else:
    print("⚠ SHAP analysis skipped for Linear Regression (not supported)")
    
    # For linear regression, show coefficients
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Coefficient': best_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\n📊 Linear Regression Coefficients:")
    print(importance_df.head(10).to_string(index=False))

print("\n✅ Analysis Complete!")
print(f"📁 Generated files:")
print(f"   - aqi_analysis.png (Exploratory Data Analysis)")
print(f"   - model_comparison.png (Model Performance)")
if best_model_name != 'Linear Regression':
    print(f"   - shap_summary.png (Feature Importance)")

print(f"\n🎯 Summary:")
print(f"   - Best Model: {best_model_name}")
print(f"   - Accuracy (R²): {results[best_model_name]['R²']:.3f}")
print(f"   - Error (RMSE): {results[best_model_name]['RMSE']:.3f}")
print(f"   - Dataset Size: {len(df)} records")
print(f"   - Features Used: {len(available_features)}")

print("\n=== Analysis Complete ===")
