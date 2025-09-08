# Air Quality Index (AQI) Analysis Project 🌍

A comprehensive machine learning project for analyzing and predicting Air Quality Index (AQI) using various environmental factors.

## 📊 Project Overview

This project analyzes air quality data from Indian cities and builds machine learning models to:
- Predict AQI based on pollutant concentrations
- Identify key factors affecting air quality
- Compare different ML algorithms' performance
- Visualize air quality trends and patterns

## 🚀 Features

- **Data Analysis**: Comprehensive EDA with visualizations
- **Multiple ML Models**: Random Forest, XGBoost, Linear Regression, SVR
- **Feature Importance**: SHAP analysis for model interpretability
- **Automated Pipeline**: Easy-to-run scripts for complete analysis
- **Visualization**: Interactive plots and charts

## 📁 Project Structure

```
proj-1/
├── 1_optimized.py          # Main analysis script (optimized)
├── 1.py                    # Original analysis script
├── run_project.bat         # Easy runner script
├── setup_environment.bat   # Environment setup script
├── city_day.csv           # Dataset (download required)
├── aqi_analysis.png       # Generated analysis plots
├── model_comparison.png   # Model performance comparison
├── shap_summary.png       # SHAP feature importance
└── README.md              # This file
```

## 🛠️ Quick Start

### Option 1: Using Batch Scripts (Windows)

1. **First-time setup:**
   ```bash
   setup_environment.bat
   ```

2. **Run analysis:**
   ```bash
   run_project.bat
   ```

### Option 2: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/aqi-analysis.git
   cd aqi-analysis
   ```

2. **Create conda environment:**
   ```bash
   conda create --prefix .conda python=3.9 -y
   ```

3. **Install dependencies:**
   ```bash
   .conda/Scripts/pip.exe install pandas numpy matplotlib seaborn scikit-learn xgboost shap
   ```

4. **Download dataset:**
   - Download `city_day.csv` from [Kaggle Air Quality Dataset](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
   - Place it in the project root directory

5. **Run analysis:**
   ```bash
   .conda/python.exe 1_optimized.py
   ```

## 📦 Dependencies

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap

## 📊 Dataset

The project uses the Air Quality Data in India dataset from Kaggle:
- **Source**: [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Features**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Target**: AQI (Air Quality Index)
- **Time Period**: 2015-2020
- **Cities**: Multiple Indian cities

## 🎯 Results

The analysis generates three main outputs:

1. **aqi_analysis.png**: Comprehensive EDA visualizations
2. **model_comparison.png**: ML model performance comparison
3. **shap_summary.png**: Feature importance analysis

### Model Performance
- **Random Forest**: Best overall performance
- **XGBoost**: Close second with good interpretability
- **Linear Regression**: Baseline model
- **SVR**: Slower but competitive accuracy

### Key Insights
- PM2.5 and PM10 are the strongest predictors of AQI
- Seasonal patterns significantly affect air quality
- Urban areas show higher pollution levels
- Chemical pollutants have varying impact across cities

## 🔬 Analysis Features

- **Exploratory Data Analysis (EDA)**
  - Missing value analysis
  - Distribution plots
  - Correlation analysis
  - Time series trends

- **Machine Learning Pipeline**
  - Data preprocessing
  - Feature engineering
  - Model training and evaluation
  - Cross-validation
  - Hyperparameter tuning

- **Model Interpretability**
  - SHAP (SHapley Additive exPlanations) analysis
  - Feature importance ranking
  - Model prediction explanations

## 🚀 Future Enhancements

- [ ] Real-time data integration
- [ ] Deep learning models (LSTM, CNN)
- [ ] Web dashboard for interactive analysis
- [ ] API for AQI predictions
- [ ] Mobile app integration
- [ ] More cities and international data

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Vicky
- **Project Link**: [https://github.com/your-username/aqi-analysis](https://github.com/your-username/aqi-analysis)

## 🙏 Acknowledgments

- Kaggle for providing the Air Quality dataset
- SHAP library for model interpretability
- Scikit-learn and XGBoost communities
- Environmental data providers and researchers

---

⭐ **Star this repository if you found it helpful!**
