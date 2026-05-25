# =============================================================================
# 03_models_group_a.py — Group A: End-to-End models (no decomposition)
#   A1: SARIMA
#   A2: XGBoost
#   A3: LSTM
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import *


# =============================================================================
# UTILITY: Multi-step recursive forecasting
# =============================================================================

def recursive_forecast(model_predict_fn, last_known: np.ndarray, horizon: int) -> np.ndarray:
    """
    Recursive (iterated) strategy: use model's own prediction as input for next step.
    Used for XGBoost multi-step forecasting.
    """
    preds  = []
    window = list(last_known)
    for _ in range(horizon):
        x    = np.array(window[-len(last_known):]).reshape(1, -1)
        pred = model_predict_fn(x)[0]
        preds.append(pred)
        window.append(pred)
    return np.array(preds)


# =============================================================================
# A1: SARIMA
# =============================================================================

def train_sarima(train_series: pd.Series, use_auto: bool = True):
    """
    Fit SARIMA model using pmdarima auto_arima for order selection.
    Falls back to manual SARIMA(1,1,1)(1,1,1,7) if auto is too slow.

    Note: period=365 is computationally expensive. We use period=7 (weekly) 
    as an approximation and add annual dummies via exogenous if needed.
    """
    try:
        import pmdarima as pm
        print("  Running auto_arima (weekly seasonal approximation, m=7)...")
        model = pm.auto_arima(
            train_series,
            start_p=1, max_p=SARIMA_MAX_P,
            start_q=1, max_q=SARIMA_MAX_Q,
            d=None, D=1,
            m=SARIMA_M_APPROX,              # weekly seasonality as proxy
            start_P=0, max_P=SARIMA_MAX_P_S,
            start_Q=0, max_Q=SARIMA_MAX_Q_S,
            seasonal=True,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_jobs=1,
        )
        print(f"  Selected order: {model.order} × {model.seasonal_order}")
        return model, "pmdarima"

    except Exception as e:
        print(f"  auto_arima failed ({e}), falling back to statsmodels SARIMA(1,1,1)(1,1,1,7)")
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return model, "statsmodels"


def sarima_forecast(model, model_type: str, horizon: int) -> np.ndarray:
    """Generate h-step ahead forecast from fitted SARIMA."""
    if model_type == "pmdarima":
        fc, _ = model.predict(n_periods=horizon, return_conf_int=True)
        return np.array(fc)
    else:
        fc = model.forecast(steps=horizon)
        return np.array(fc)


def sarima_rolling_forecast(train: pd.Series, val_test: pd.Series,
                             horizon: int,
                             refit_every: int = None) -> np.ndarray:
    """
    Walk-forward SARIMA forecast — optimised for personal laptops.

    Strategy (3 layers of optimisation):
    ─────────────────────────────────────────────────────────────
    1. Full refit (auto_arima) only every `refit_every` days.
       Default: SARIMA_REFIT_EVERY from config (30 days).
       → ~12 refits over 365-day test set instead of 365.

    2. Between refits: use pmdarima model.update([obs]) to
       assimilate new data without re-searching (p,d,q) order.
       → Near-instant state update per day.

    3. m=7 (weekly seasonal proxy) for all SARIMA fits.
       Annual seasonality is captured upstream by STL.
       → Each refit takes ~30–60s instead of minutes.

    Academic justification:
    "Following Hyndman & Athanasopoulos (2018), we employ a
    batch walk-forward strategy with full model re-estimation
    every 30 observations, using online parameter updates
    (pmdarima update()) for intermediate steps. This balances
    computational tractability with forecast accuracy."
    ─────────────────────────────────────────────────────────────
    """
    import pmdarima as pm

    if refit_every is None:
        refit_every = SARIMA_REFIT_EVERY

    preds       = []
    history     = list(train.values)
    history_idx = list(train.index)
    series_vals = list(val_test.values)
    series_idx  = list(val_test.index)
    n           = len(series_vals)
    model       = None   # pmdarima ARIMA object

    step = 0
    while step < n:
        # ── Decide: full refit or just update? ──────────────────
        need_refit = (model is None) or (step % refit_every == 0)

        if need_refit:
            current_series = pd.Series(history,
                index=pd.DatetimeIndex(history_idx))
            print(f"    [SARIMA] Full refit at step {step}/{n} "
                  f"(history={len(history)} obs)...")
            model, _ = train_sarima(current_series, use_auto=(step == 0))
            # Wrap statsmodels fallback into pmdarima-compatible object
            # so .update() is always available
            if not hasattr(model, "update"):
                # statsmodels SARIMAXResults — wrap minimally
                model = _wrap_statsmodels(model, current_series)

        # ── Multi-step forecast ──────────────────────────────────
        h  = min(horizon, n - step)
        fc = model.predict(n_periods=h)
        preds.extend(np.array(fc).tolist())

        # ── Assimilate observed values one by one ────────────────
        for fi in range(h):
            obs = series_vals[step + fi]
            model.update([obs])            # O(1) — no grid search
            history.append(obs)
            history_idx.append(series_idx[step + fi])

        step += h
        if step % 60 == 0 or step >= n:
            print(f"    [SARIMA] walk-forward progress: {step}/{n}")

    return np.array(preds[:n])


def _wrap_statsmodels(sm_result, train_series: pd.Series):
    """
    Thin wrapper so a statsmodels SARIMAXResults behaves like
    a pmdarima ARIMA for .predict() and .update() calls.
    Falls back gracefully — refit from scratch on .update().
    """
    import pmdarima as pm

    class _CompatModel:
        def __init__(self, res, series):
            self._res    = res
            self._series = list(series.values)
            self._order  = res.model.order
            self._sorder = res.model.seasonal_order

        def predict(self, n_periods=1):
            fc = self._res.forecast(steps=n_periods)
            return np.array(fc)

        def update(self, new_obs):
            self._series.extend(new_obs)
            # Lightweight refit on updated series
            s = pd.Series(self._series)
            try:
                new_model = pm.ARIMA(
                    order=self._order,
                    seasonal_order=self._sorder,
                    suppress_warnings=True,
                ).fit(s)
                self._res = new_model
                # Make predict/update delegate to pmdarima directly
                self.predict = new_model.predict
                self.update  = new_model.update
            except Exception:
                pass  # keep old model if refit fails

    return _CompatModel(sm_result, train_series)


# =============================================================================
# A2: XGBoost
# =============================================================================

def build_xgb_features(df_feat: pd.DataFrame, target: str = TARGET):
    """Extract X, y from feature-engineered DataFrame."""
    feature_cols = [c for c in df_feat.columns if c != target
                    and not c.startswith("doy") and c != "month" and c != "week"]
    # Include cyclic calendar features
    feature_cols = [c for c in df_feat.columns if c != target]
    X = df_feat[feature_cols].values
    y = df_feat[target].values
    return X, y, feature_cols


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray,   y_val: np.ndarray):
    """Train XGBoost with early stopping on validation set."""
    from xgboost import XGBRegressor
    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=XGB_EARLY_STOPPING)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"  XGBoost trained. Best iteration: {model.best_iteration}")
    return model


def xgb_direct_forecast(model, df_feat: pd.DataFrame, target_dates: pd.DatetimeIndex,
                          horizon: int, df_full: pd.DataFrame) -> np.ndarray:
    """
    Direct multi-step strategy: train a separate model per horizon h.
    Avoids error accumulation in recursive approach.
    Returns predictions for all target_dates.
    """
    from xgboost import XGBRegressor
    from importlib import import_module
    dp = import_module("data_prep")

    preds = []
    for date in target_dates:
        # Build feature vector using data up to (date - horizon) to avoid leakage
        cutoff = date - pd.Timedelta(days=horizon)
        hist = df_full.loc[:cutoff]
        if len(hist) < max(LAG_DAYS) + max(ROLLING_WINDOWS):
            preds.append(np.nan)
            continue
        feat_row = dp.make_features(hist).iloc[-1]
        feat_cols = [c for c in feat_row.index if c != TARGET]
        X = feat_row[feat_cols].values.reshape(1, -1)
        # Ensure same feature count
        if X.shape[1] != model.n_features_in_:
            preds.append(np.nan)
            continue
        preds.append(model.predict(X)[0])

    return np.array(preds)


def get_feature_importance(model, feature_cols: list, save_dir: str = OUTPUT_DIR, tag: str = "A2"):
    """Plot XGBoost feature importance."""
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot.barh(ax=ax, color="#DD8452")
    ax.set_title(f"XGBoost feature importance (top 20) — {tag}", fontsize=12)
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_fi_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# A3: LSTM
# =============================================================================

def build_lstm_sequences(series: np.ndarray, lookback: int = LSTM_LOOKBACK):
    """Convert 1D time series to supervised sequences for LSTM."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback: i])
        y.append(series[i])
    return np.array(X)[..., np.newaxis], np.array(y)  # (N, lookback, 1)


def build_lstm_model(lookback: int = LSTM_LOOKBACK,
                      units: list = LSTM_UNITS,
                      dropout: float = LSTM_DROPOUT):
    """Build a stacked LSTM model."""
    import tensorflow as tf
    tf.random.set_seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(lookback, 1)))
    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        model.add(tf.keras.layers.LSTM(u, return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    return model


def train_lstm(train_series: np.ndarray, val_series: np.ndarray,
               scaler=None, lookback: int = LSTM_LOOKBACK, tag: str = "A3"):
    """
    Normalise, build sequences, fit LSTM with early stopping.
    Returns fitted model + scaler.
    """
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))

    all_data = np.concatenate([train_series, val_series])
    scaler.fit(train_series.reshape(-1, 1))

    train_sc = scaler.transform(train_series.reshape(-1, 1)).flatten()
    val_sc   = scaler.transform(val_series.reshape(-1, 1)).flatten()

    X_train, y_train = build_lstm_sequences(train_sc, lookback)
    X_val,   y_val   = build_lstm_sequences(np.concatenate([train_sc[-lookback:], val_sc]), lookback)

    model = build_lstm_model(lookback)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
        callbacks=cb, verbose=0,
    )
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    print(f"  LSTM ({tag}) trained. Best epoch: {best_epoch}/{LSTM_EPOCHS}")
    return model, scaler


def lstm_forecast(model, history_scaled: np.ndarray, horizon: int,
                   scaler, lookback: int = LSTM_LOOKBACK) -> np.ndarray:
    """Recursive LSTM forecast for h steps ahead."""
    import tensorflow as tf
    window = list(history_scaled[-lookback:])
    preds  = []
    for _ in range(horizon):
        x    = np.array(window[-lookback:]).reshape(1, lookback, 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        window.append(pred)
    preds_orig = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_orig


# =============================================================================
# MAIN — quick sanity check
# =============================================================================

if __name__ == "__main__":
    print("Group A models module loaded. Import and call individual functions.")
    print("  train_sarima(), train_xgboost(), train_lstm()")
    print("  Run 05_run_pipeline.py for full evaluation.")
