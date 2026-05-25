# =============================================================================
# 04_models_group_b.py — Group B: Decompose-then-Forecast (STL first)
#   B1: STL + SARIMA  (trend + residual modeled by SARIMA)
#   B2: STL + XGBoost (residual modeled by XGBoost, seasonal reused)
#   B3: STL + LSTM    (residual modeled by LSTM, seasonal reused)
#
# Core equation:
#   ŷ(t+h) = T̂(t+h) + S_pattern[doy(t+h)] + R̂(t+h)
#            └─ SARIMA/ETS ─┘  └─── fixed ───┘  └─ ML/DL ─┘
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import *
from models_group_a import (
    train_sarima, sarima_forecast,
    train_xgboost, build_xgb_features,
    train_lstm, lstm_forecast, build_lstm_sequences,
)


# =============================================================================
# SHARED UTILITY: Rebuild seasonal value for a future date
# =============================================================================

def seasonal_lookup(seasonal_pattern: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Look up the average seasonal component for a list of future dates.
    seasonal_pattern: Series indexed by day-of-year (1–366).
    """
    values = []
    for d in dates:
        doy = d.dayofyear
        if doy in seasonal_pattern.index:
            values.append(seasonal_pattern[doy])
        else:
            # Leap year day 366 — fall back to day 365
            values.append(seasonal_pattern.get(365, 0.0))
    return np.array(values)


# =============================================================================
# B1: STL + SARIMA
# =============================================================================

def train_stl_sarima(train_series: pd.Series, components: dict):
    """
    Fit SARIMA on the trend component and separately on the residual component.
    
    Two-model approach:
    - SARIMA_trend: models the slow-moving trend
    - SARIMA_resid: models remaining autocorrelation in residuals
    """
    trend  = components["trend"]
    resid  = components["residual"]

    print("  Fitting SARIMA on trend...")
    model_trend, mtype_trend = train_sarima(trend, use_auto=True)

    print("  Fitting SARIMA on residual...")
    model_resid, mtype_resid = train_sarima(resid, use_auto=True)

    return {
        "model_trend": model_trend, "mtype_trend": mtype_trend,
        "model_resid": model_resid, "mtype_resid": mtype_resid,
    }


def stl_sarima_forecast(models: dict, seasonal_pattern: pd.Series,
                         future_dates: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """
    Single-call forecast for B1 (used inside rolling loop in run_pipeline).
    ŷ = T̂ (SARIMA on trend) + S (seasonal pattern) + R̂ (SARIMA on resid)
    """
    h     = min(horizon, len(future_dates))
    T_hat = models["model_trend"].predict(n_periods=h)
    R_hat = models["model_resid"].predict(n_periods=h)
    S_hat = seasonal_lookup(seasonal_pattern, future_dates[:h])
    return np.array(T_hat) + S_hat + np.array(R_hat)


def stl_sarima_rolling_forecast(models: dict, components_train: dict,
                                  seasonal_pattern: pd.Series,
                                  test_series: pd.Series, horizon: int,
                                  refit_every: int = None) -> np.ndarray:
    """
    Walk-forward forecast for B1: STL + SARIMA.
    Uses same batch-refit + pmdarima.update() strategy as Group A SARIMA.

    After each observed window:
    - model_trend.update([obs_trend])   ← online update, no grid search
    - model_resid.update([obs_resid])   ← same
    Full refit every `refit_every` days (default 30).
    """
    from config import SARIMA_REFIT_EVERY
    from models_group_a import train_sarima

    if refit_every is None:
        refit_every = SARIMA_REFIT_EVERY

    preds        = []
    n            = len(test_series)
    model_trend  = models["model_trend"]
    model_resid  = models["model_resid"]

    # Keep running history of decomposed components
    trend_hist   = list(components_train["trend"].values)
    resid_hist   = list(components_train["residual"].values)
    trend_idx    = list(components_train["trend"].index)

    step = 0
    while step < n:
        h            = min(horizon, n - step)
        future_dates = test_series.index[step: step + h]

        # ── Full refit every 30 days ─────────────────────────────
        if step > 0 and step % refit_every == 0:
            print(f"    [B1-SARIMA] Batch refit at step {step}/{n}...")
            s_trend = pd.Series(trend_hist, index=pd.DatetimeIndex(trend_idx))
            s_resid = pd.Series(resid_hist, index=pd.DatetimeIndex(trend_idx))
            model_trend, _ = train_sarima(s_trend, use_auto=False)
            model_resid, _ = train_sarima(s_resid, use_auto=False)

        # ── Forecast ─────────────────────────────────────────────
        T_hat = np.array(model_trend.predict(n_periods=h))
        R_hat = np.array(model_resid.predict(n_periods=h))
        S_hat = seasonal_lookup(seasonal_pattern, future_dates)
        y_hat = T_hat + S_hat + R_hat
        preds.extend(y_hat.tolist())

        # ── Online update with observed decomposed values ─────────
        # Approximate: decompose observed temp into T+S+R on the fly
        for fi in range(h):
            obs        = test_series.iloc[step + fi]
            s_val      = seasonal_lookup(seasonal_pattern,
                             pd.DatetimeIndex([future_dates[fi]]))[0]
            # Trend approximation: rolling mean of last 30 obs
            t_val      = np.mean(trend_hist[-30:])
            r_val      = obs - t_val - s_val
            model_trend.update([t_val])
            model_resid.update([r_val])
            trend_hist.append(t_val)
            resid_hist.append(r_val)
            trend_idx.append(future_dates[fi])

        step += h

    return np.array(preds[:n])


# =============================================================================
# B2: STL + XGBoost
# =============================================================================

def make_residual_features(residual: pd.Series) -> pd.DataFrame:
    """
    Feature matrix for modeling STL residuals with XGBoost.
    Uses lag and rolling features on the residual series.
    """
    df = pd.DataFrame({"residual": residual})

    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["residual"].shift(lag)

    for w in [7, 14]:
        df[f"rolling_mean_{w}"] = df["residual"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["residual"].shift(1).rolling(w).std()

    # Calendar (same sin/cos encoding)
    df["doy_sin"]   = np.sin(2 * np.pi * residual.index.dayofyear / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * residual.index.dayofyear / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * residual.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * residual.index.month / 12)

    return df.dropna()


def train_stl_xgboost(components_train: dict, components_val: dict,
                       train_series: pd.Series, val_series: pd.Series):
    """
    Train XGBoost on STL residuals.
    Separately train SARIMA on the trend.
    """
    from xgboost import XGBRegressor

    resid_train = components_train["residual"]
    resid_val   = components_val["residual"]

    df_train = make_residual_features(resid_train)
    df_val   = make_residual_features(resid_val)

    X_train = df_train.drop(columns=["residual"]).values
    y_train = df_train["residual"].values
    X_val   = df_val.drop(columns=["residual"]).values
    y_val   = df_val["residual"].values

    model_resid = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=XGB_EARLY_STOPPING)
    model_resid.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)], verbose=False)
    print(f"  XGBoost (residual) trained. Best iter: {model_resid.best_iteration}")

    # SARIMA on trend
    trend_train = components_train["trend"]
    print("  Fitting SARIMA on trend...")
    model_trend, mtype_trend = train_sarima(trend_train, use_auto=True)

    return {
        "model_resid": model_resid,
        "model_trend": model_trend, "mtype_trend": mtype_trend,
        "feature_cols": list(df_train.drop(columns=["residual"]).columns),
    }


def stl_xgb_rolling_forecast(models: dict, components_train: dict,
                               seasonal_pattern: pd.Series,
                               test_series: pd.Series, horizon: int) -> np.ndarray:
    """
    Walk-forward forecast using STL + XGBoost.
    For each step: predict trend (SARIMA), look up seasonal, predict residual (XGB).
    """
    preds          = []
    resid_history  = list(components_train["residual"].values)
    resid_index    = list(components_train["residual"].index)
    n              = len(test_series)
    model_resid    = models["model_resid"]
    model_trend    = models["model_trend"]
    mtype_trend    = models["mtype_trend"]
    feature_cols   = models["feature_cols"]

    step = 0
    while step < n:
        h = min(horizon, n - step)
        future_dates = test_series.index[step: step + h]

        # Trend forecast
        T_hat = sarima_forecast(model_trend, mtype_trend, h)

        # Seasonal lookup
        S_hat = seasonal_lookup(seasonal_pattern, future_dates)

        # Residual forecast — build feature vector from recent residuals
        resid_series = pd.Series(resid_history, index=resid_index)
        df_feat = make_residual_features(resid_series)
        R_hats = []
        for fi in range(h):
            if len(df_feat) > 0:
                X = df_feat.iloc[-1][feature_cols].values.reshape(1, -1)
                r_hat = model_resid.predict(X)[0]
            else:
                r_hat = 0.0
            R_hats.append(r_hat)
            # Update residual history with observed minus expected
            observed = test_series.iloc[step + fi]
            true_resid = observed - (T_hat[fi] + S_hat[fi])
            resid_history.append(true_resid)
            resid_index.append(future_dates[fi])
            resid_series = pd.Series(resid_history, index=resid_index)
            df_feat = make_residual_features(resid_series)

        y_hat = T_hat + S_hat + np.array(R_hats)
        preds.extend(y_hat.tolist())
        step += h

    return np.array(preds[:n])


# =============================================================================
# B3: STL + LSTM
# =============================================================================

def train_stl_lstm(components_train: dict, components_val: dict):
    """
    Train LSTM on STL residuals + SARIMA on trend.
    LSTM learns non-linear patterns left in the residual.
    """
    resid_train = components_train["residual"].values
    resid_val   = components_val["residual"].values

    model_lstm, scaler = train_lstm(resid_train, resid_val, tag="B3-residual")

    # SARIMA on trend
    trend_train = components_train["trend"]
    print("  Fitting SARIMA on trend...")
    model_trend, mtype_trend = train_sarima(trend_train, use_auto=True)

    return {
        "model_lstm":  model_lstm,
        "scaler":      scaler,
        "model_trend": model_trend,
        "mtype_trend": mtype_trend,
    }


def stl_lstm_rolling_forecast(models: dict, components_train: dict,
                                seasonal_pattern: pd.Series,
                                test_series: pd.Series, horizon: int) -> np.ndarray:
    """
    Walk-forward forecast using STL + LSTM.
    Same structure as STL+XGB but LSTM models the residual.
    """
    from sklearn.preprocessing import MinMaxScaler

    preds         = []
    model_lstm    = models["model_lstm"]
    scaler        = models["scaler"]
    model_trend   = models["model_trend"]
    mtype_trend   = models["mtype_trend"]

    resid_history = list(components_train["residual"].values)
    n             = len(test_series)
    lookback      = LSTM_LOOKBACK

    step = 0
    while step < n:
        h = min(horizon, n - step)
        future_dates = test_series.index[step: step + h]

        # Trend forecast
        T_hat = sarima_forecast(model_trend, mtype_trend, h)

        # Seasonal lookup
        S_hat = seasonal_lookup(seasonal_pattern, future_dates)

        # Residual: LSTM recursive
        hist_sc = scaler.transform(
            np.array(resid_history).reshape(-1, 1)).flatten()
        R_hat = lstm_forecast(model_lstm, hist_sc, h, scaler, lookback)

        y_hat = T_hat + S_hat + R_hat
        preds.extend(y_hat.tolist())

        # Update residual history
        for fi in range(h):
            if step + fi < n:
                observed   = test_series.iloc[step + fi]
                true_resid = observed - (T_hat[fi] + S_hat[fi])
                resid_history.append(true_resid)

        step += h
        if step % 30 == 0 or step >= n:
            print(f"    STL+LSTM walk-forward: {step}/{n} steps done")

    return np.array(preds[:n])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Group B models module loaded.")
    print("  Functions: train_stl_sarima, train_stl_xgboost, train_stl_lstm")
    print("  Run 05_run_pipeline.py for full evaluation.")
