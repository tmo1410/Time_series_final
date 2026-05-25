# =============================================================================
# config.py — Central configuration for Hanoi Temperature Forecasting Study
# Research: STL Decomposition vs End-to-End forecasting
# =============================================================================

import os

# --- Paths ---
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "hanoi_weather.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# --- Target variable ---
TARGET = "temp"           # daily mean temperature (°C)
DATE_COL = "datetime"

# --- Train / Val / Test split (chronological, no leakage) ---
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
# Test: 2024-01-01 → 2024-12-31 (365 ngày, đủ 4 mùa)

# --- SARIMA walk-forward strategy ---
# Refit every N days (not every step) to avoid 365x refitting
# pmdarima .update() is used between batch refits
SARIMA_REFIT_EVERY = 30     # refit full model every 30 days
# → ~12 actual refits over 365-day test set (vs 365 refits before)
# Justification: temperature dynamics stable within 30-day windows

# --- Forecast horizons (days ahead) ---
HORIZONS = [1, 3, 7]

# --- STL parameters ---
STL_PERIOD = 365          # annual seasonality
STL_ROBUST = True         # downweight outliers via iterative reweighting

# --- SARIMA search space (for auto_arima) ---
SARIMA_MAX_P = 3
SARIMA_MAX_Q = 3
SARIMA_MAX_P_S = 2        # seasonal P
SARIMA_MAX_Q_S = 2        # seasonal Q
SARIMA_M = 365            # seasonal period (approx; use 7 for weekly if too slow)
SARIMA_M_APPROX = 7       # faster approximation: weekly seasonality as proxy

# --- XGBoost parameters ---
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
XGB_EARLY_STOPPING = 50

# --- LSTM parameters ---
LSTM_LOOKBACK   = 30      # sequence length (days)
LSTM_UNITS      = [64, 32]
LSTM_DROPOUT    = 0.2
LSTM_EPOCHS     = 100
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE   = 10      # early stopping patience
LSTM_SEED       = 42

# --- Feature engineering: lag days ---
LAG_DAYS = [1, 2, 3, 7, 14, 30, 365]

# --- Rolling windows ---
ROLLING_WINDOWS = [7, 14, 30]

# --- Evaluation ---
DM_LOSS = "squared"       # "squared" or "absolute"
DM_H    = 1               # forecast horizon for DM test (steps ahead)

# --- Plotting ---
PLOT_LAST_N_DAYS = 90     # show last N days in forecast vs actual plot
FIGSIZE_WIDE  = (14, 5)
FIGSIZE_SQUARE = (10, 8)
PALETTE = {
    "SARIMA":     "#4C72B0",
    "XGBoost":    "#DD8452",
    "LSTM":       "#55A868",
    "STL+SARIMA": "#C44E52",
    "STL+XGBoost":"#8172B2",
    "STL+LSTM":   "#937860",
    "Naive":      "#BBBBBB",
}
