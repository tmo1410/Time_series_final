# Time Series Forecasting: Hanoi Temperature Prediction

Name: Nguyen Thanh Mo

ID: 11230571

Class: DSEB 65B

Subject: Time Series

## Overview

This project implements a comprehensive time series forecasting study comparing **STL Decomposition-based models** versus **End-to-End forecasting models** for daily temperature prediction in Hanoi. The research evaluates 6 different forecasting models across 3 prediction horizons (1, 3, and 7 days ahead).

## 📊 Research Design

### Models Comparison

**Group A - End-to-End Models (Direct forecasting):**
- SARIMA (Seasonal ARIMA)
- XGBoost
- LSTM (Deep Learning)

**Group B - STL Decomposition-Based Models:**
- SARIMA on STL trend
- XGBoost on STL trend + seasonal
- LSTM on STL decomposed components

### Forecast Horizons

- **1-day ahead** - Short-term prediction
- **3-day ahead** - Medium-term prediction  
- **7-day ahead** - Week-ahead prediction

### Data

- **Dataset:** Hanoi weather data (daily measurements)
- **Target variable:** `temp` (daily mean temperature in °C)
- **Time split:**
  - Training: 2021-01-01 to 2022-12-31
  - Validation: 2023-01-01 to 2023-12-31
  - Test: 2024-01-01 to 2024-12-31 (365 days, covering all seasons)

## 🏗️ Project Structure

```
.
├── config.py                  # Central configuration & hyperparameters
├── data_prep.py              # Data loading, cleaning, feature engineering
├── stl_decomposition.py      # STL decomposition & analysis
├── models_group_a.py         # End-to-end forecasting models
├── models_group_b.py         # STL-based forecasting models
├── evaluation.py             # Performance metrics & Diebold-Mariano test
├── run_pipeline.py           # Master orchestrator for full pipeline
├── requirements.txt          # Python dependencies
├── data/
│   └── hanoi_weather.csv     # Raw weather data
└── outputs/                  # Generated results, plots, and metrics
```

## 🚀 Quick Start

### Installation

1. **Clone and navigate to project:**
   ```bash
   cd /workspaces/Time_series_final
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

**Full pipeline execution:**
```bash
python run_pipeline.py
```

**Fast mode (skip LSTM training):**
```bash
python run_pipeline.py --fast
```

**Single horizon test (e.g., 1-day ahead only):**
```bash
python run_pipeline.py --horizon 1
```

## 📋 Pipeline Steps

The complete pipeline executes the following steps:

1. **Data Preparation** (`data_prep.py`)
   - Load and clean weather data
   - Perform stationarity tests (ADF, KPSS)
   - Create lag features and technical indicators
   - Generate EDA plots

2. **STL Decomposition** (`stl_decomposition.py`)
   - Decompose time series into Trend, Seasonal, and Residual components
   - Analyze residual diagnostics
   - Extract seasonal patterns

3. **Model Training** (`models_group_a.py`, `models_group_b.py`)
   - **Group A:** Train SARIMA, XGBoost, and LSTM on original series
   - **Group B:** Train models on STL-decomposed components
   - Walk-forward validation for realistic performance assessment
   - SARIMA refit strategy: Every 30 days (~12 refits over 365-day test set)

4. **Evaluation** (`evaluation.py`)
   - Calculate metrics: MAE, RMSE, MAPE, R²
   - Perform Diebold-Mariano (DM) statistical tests
   - Compare model forecasts systematically
   - Generate comparison tables and visualizations

5. **Results Generation**
   - Output files in `outputs/` directory
   - Plots: decomposition, forecasts, residuals, comparisons
   - CSV tables with all metrics and DM test results

## 🔧 Key Features

### Configuration

Edit `config.py` to customize:
- **Forecast horizons:** `HORIZONS = [1, 3, 7]`
- **STL seasonality period:** `STL_PERIOD = 365`
- **SARIMA refit frequency:** `SARIMA_REFIT_EVERY = 30`
- **Model hyperparameters:** XGBoost, LSTM, and SARIMA settings

### Data Processing

The `data_prep.py` module includes:
- Stationarity testing (ADF, KPSS tests)
- Feature engineering (lags, rolling statistics)
- Chronological train-val-test split (no data leakage)
- EDA visualizations

### Statistical Testing

The `evaluation.py` module performs:
- **Diebold-Mariano (DM) Test:** Tests if one forecast is significantly better than another
- **Multiple error metrics:** MAE, RMSE, MAPE, R²
- **Directional accuracy:** Percentage of correct direction predictions

## 📦 Dependencies

Core libraries:
- **Data & Numerics:** pandas, numpy, scipy
- **Time Series:** statsmodels, pmdarima, arch
- **ML/DL:** scikit-learn, xgboost, tensorflow
- **Visualization:** matplotlib, seaborn

See `requirements.txt` for specific versions.

## 📈 Expected Outputs

After running the pipeline, the `outputs/` directory contains:

- **Plots:**
  - STL decomposition visualization
  - Forecast comparison charts
  - Residual diagnostics
  - Multi-horizon prediction results

- **Data:**
  - `results_metrics.csv` - Performance metrics for all models/horizons
  - `dm_test_results.csv` - Diebold-Mariano test comparisons

## 🧪 Testing & Debugging

- **Check data:** Run `data_prep.load_and_clean()` to inspect data quality
- **Debug STL:** Review STL decomposition plots in outputs
- **Model validation:** Check fitted model summaries in logs
- **Fast testing:** Use `--fast` flag to skip slow LSTM training

## 📚 Research Notes

- **STL Decomposition**: Uses annual seasonality (365-day period) to capture weather patterns
- **Walk-Forward Validation**: Prevents look-ahead bias in time series evaluation
- **SARIMA Refit Strategy**: Balance between model adaptability and computational efficiency
- **Naive Baseline**: Persistence forecast (shift by horizon) for baseline comparison

## 🤝 Contributing

For modifications:
1. Update configuration in `config.py`
2. Modify corresponding model/evaluation module
3. Run `run_pipeline.py` to validate changes
4. Check outputs directory for results

## 📝 License

Research project - Internal use

---

**Last Updated:** 2024
**Framework Version:** TensorFlow 2.13+, scikit-learn 1.3+, statsmodels 0.14+
