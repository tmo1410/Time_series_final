# =============================================================================
# 06_run_pipeline.py — Master orchestrator
#
# Runs the complete research pipeline:
#   1. Load & prep data
#   2. STL decomposition
#   3. Train all 6 models (Group A + B) across 3 horizons
#   4. Evaluate with metrics + DM test
#   5. Generate all plots and results table
#
# Usage:
#   python 06_run_pipeline.py                 # full run
#   python 06_run_pipeline.py --fast          # skip LSTM (faster for testing)
#   python 06_run_pipeline.py --horizon 1     # single horizon
# =============================================================================

import sys
import os
import argparse
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add current directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
    
# Lazy imports — avoids TF loading if only using SARIMA/XGB
def _import_all():
    global dp, stl_mod, ga, gb, ev
    import importlib
    dp      = importlib.import_module("data_prep")
    stl_mod = importlib.import_module("stl_decomposition")
    ga      = importlib.import_module("models_group_a")
    gb      = importlib.import_module("models_group_b")
    ev      = importlib.import_module("evaluation")


# =============================================================================
# NAIVE BASELINE: persistence forecast (ŷ(t+h) = y(t))
# =============================================================================

def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Persistence: forecast = last known value (shift by horizon)."""
    return series.shift(horizon).dropna().values


# =============================================================================
# STEP 1: DATA
# =============================================================================

def step_data(args):
    print("\n" + "="*60)
    print("  STEP 1: Data preparation")
    print("="*60)
    df = dp.load_and_clean()
    train, val, test = dp.split_data(df)
    dp.run_stationarity_tests(train[TARGET])
    dp.plot_eda(df)
    df_feat = dp.make_features(df)
    return df, train, val, test, df_feat


# =============================================================================
# STEP 2: STL DECOMPOSITION
# =============================================================================

def step_stl(train, val):
    print("\n" + "="*60)
    print("  STEP 2: STL Decomposition")
    print("="*60)

    components_train = stl_mod.fit_stl(train[TARGET])
    stl_mod.plot_decomposition(train[TARGET], components_train)
    stl_mod.diagnose_residuals(components_train["residual"])
    seasonal_pattern = stl_mod.extract_seasonal_pattern(components_train)

    # Also decompose val (needed for Group B val-set residual model training)
    print("\nDecomposing validation set...")
    components_val = stl_mod.fit_stl(val[TARGET])

    return components_train, components_val, seasonal_pattern


# =============================================================================
# STEP 3: TRAIN MODELS
# =============================================================================

def step_train(train, val, df, df_feat, components_train, components_val,
               seasonal_pattern, fast_mode=False):
    print("\n" + "="*60)
    print("  STEP 3: Training all models")
    print("="*60)

    models = {}

    # ---- A1: SARIMA ----
    print("\n[A1] SARIMA (end-to-end)...")
    t0 = time.time()
    models["A1_sarima"], models["A1_sarima_type"] = ga.train_sarima(train[TARGET])
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- A2: XGBoost ----
    print("\n[A2] XGBoost (end-to-end)...")
    t0 = time.time()
    train_feat = df_feat.loc[train.index.min(): train.index.max()]
    val_feat   = df_feat.loc[val.index.min():   val.index.max()]
    X_tr, y_tr, feat_cols = ga.build_xgb_features(train_feat)
    X_va, y_va, _         = ga.build_xgb_features(val_feat)
    models["A2_xgb"]      = ga.train_xgboost(X_tr, y_tr, X_va, y_va)
    models["A2_feat_cols"] = feat_cols
    ga.get_feature_importance(models["A2_xgb"], feat_cols, tag="A2")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- A3: LSTM ----
    if not fast_mode:
        print("\n[A3] LSTM (end-to-end)...")
        t0 = time.time()
        models["A3_lstm"], models["A3_scaler"] = ga.train_lstm(
            train[TARGET].values, val[TARGET].values, tag="A3")
        print(f"  Done in {time.time()-t0:.1f}s")
    else:
        print("\n[A3] LSTM skipped (--fast mode)")
        models["A3_lstm"] = None

    # ---- B1: STL + SARIMA ----
    print("\n[B1] STL + SARIMA...")
    t0 = time.time()
    models["B1_stl_sarima"] = gb.train_stl_sarima(train[TARGET], components_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- B2: STL + XGBoost ----
    print("\n[B2] STL + XGBoost...")
    t0 = time.time()
    models["B2_stl_xgb"] = gb.train_stl_xgboost(
        components_train, components_val, train[TARGET], val[TARGET])
    print(f"  Done in {time.time()-t0:.1f}s")

    # ---- B3: STL + LSTM ----
    if not fast_mode:
        print("\n[B3] STL + LSTM...")
        t0 = time.time()
        models["B3_stl_lstm"] = gb.train_stl_lstm(components_train, components_val)
        print(f"  Done in {time.time()-t0:.1f}s")
    else:
        print("\n[B3] STL+LSTM skipped (--fast mode)")
        models["B3_stl_lstm"] = None

    return models


# =============================================================================
# STEP 4: GENERATE FORECASTS
# =============================================================================

def step_forecast(models, train, val, test, df, df_feat,
                   components_train, seasonal_pattern, fast_mode=False):
    print("\n" + "="*60)
    print("  STEP 4: Generating forecasts")
    print("="*60)

    # Combined train+val series for final test evaluation
    train_val = pd.concat([train, val])
    target_test = test[TARGET]

    all_preds = {
        "SARIMA":     {},
        "XGBoost":    {},
        "STL+SARIMA": {},
        "STL+XGBoost":{},
    }
    if not fast_mode:
        all_preds["LSTM"]     = {}
        all_preds["STL+LSTM"] = {}

    for h in HORIZONS:
        print(f"\n  Horizon t+{h}...")

        # --- Naive baseline ---
        # Stored separately, used only for skill score

        # --- A1: SARIMA ---
        print(f"    [A1] SARIMA batch walk-forward (refit every {SARIMA_REFIT_EVERY}d)...")
        sarima_preds = ga.sarima_rolling_forecast(
            train_val[TARGET], target_test, horizon=h,
            refit_every=SARIMA_REFIT_EVERY)
        all_preds["SARIMA"][h] = sarima_preds

        # --- A2: XGBoost ---
        print(f"    [A2] XGBoost direct forecast...")
        xgb_preds = ga.xgb_direct_forecast(
            models["A2_xgb"], df_feat, target_test.index, horizon=h, df_full=df)
        all_preds["XGBoost"][h] = xgb_preds

        # --- A3: LSTM ---
        if not fast_mode and models["A3_lstm"] is not None:
            print(f"    [A3] LSTM forecast...")
            hist_sc = models["A3_scaler"].transform(
                train_val[TARGET].values.reshape(-1, 1)).flatten()
            # Walk-forward for test set
            lstm_preds = []
            history_sc = list(hist_sc)
            n = len(target_test)
            step = 0
            while step < n:
                hh = min(h, n - step)
                fc = ga.lstm_forecast(
                    models["A3_lstm"], np.array(history_sc),
                    hh, models["A3_scaler"])
                lstm_preds.extend(fc.tolist())
                history_sc.extend(
                    models["A3_scaler"].transform(
                        target_test.values[step:step+hh].reshape(-1,1)).flatten().tolist())
                step += hh
            all_preds["LSTM"][h] = np.array(lstm_preds[:n])

        # --- B1: STL + SARIMA ---
        print(f"    [B1] STL+SARIMA batch walk-forward (refit every {SARIMA_REFIT_EVERY}d)...")
        b1_preds = gb.stl_sarima_rolling_forecast(
            models["B1_stl_sarima"], components_train,
            seasonal_pattern, target_test, h,
            refit_every=SARIMA_REFIT_EVERY)
        all_preds["STL+SARIMA"][h] = b1_preds

        # --- B2: STL + XGBoost ---
        print(f"    [B2] STL+XGBoost rolling forecast...")
        b2_preds = gb.stl_xgb_rolling_forecast(
            models["B2_stl_xgb"], components_train, seasonal_pattern, target_test, h)
        all_preds["STL+XGBoost"][h] = b2_preds

        # --- B3: STL + LSTM ---
        if not fast_mode and models["B3_stl_lstm"] is not None:
            print(f"    [B3] STL+LSTM rolling forecast...")
            b3_preds = gb.stl_lstm_rolling_forecast(
                models["B3_stl_lstm"], components_train, seasonal_pattern, target_test, h)
            all_preds["STL+LSTM"][h] = b3_preds

    return all_preds, target_test


# =============================================================================
# STEP 5: EVALUATE & VISUALISE
# =============================================================================

def step_evaluate(all_preds, target_test, fast_mode=False):
    print("\n" + "="*60)
    print("  STEP 5: Evaluation & Visualisation")
    print("="*60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_true_dict = {}
    naive_dict  = {}

    for h in HORIZONS:
        n = min([len(v[h]) for v in all_preds.values() if h in v])
        y_true_dict[h] = target_test.values[:n]
        # Naive: persistence — predict last known value
        naive_dict[h] = np.array([target_test.values[max(0, i - h)]
                                    for i in range(n)])

    # Compile metrics table
    df_results = ev.compile_results(all_preds, y_true_dict, naive_dict)
    print("\n" + "="*60)
    print("  RESULTS TABLE")
    print("="*60)
    print(df_results.to_string())
    df_results.to_csv(os.path.join(OUTPUT_DIR, "results_table.csv"))
    print(f"\nSaved: results_table.csv")

    # Plots
    for metric in ["RMSE", "MAE", "MAPE"]:
        ev.plot_metrics_heatmap(df_results, metric)

    ev.plot_horizon_degradation(df_results, "RMSE")
    ev.plot_horizon_degradation(df_results, "MAE")

    for h in HORIZONS:
        ev.plot_forecast_vs_actual(target_test, all_preds, h)
        ev.plot_error_distribution(y_true_dict[h], all_preds, h)

        # DM test matrix
        h_preds = {m: v[h] for m, v in all_preds.items() if h in v}
        n = min(len(v) for v in h_preds.values())
        y_true_trimmed = y_true_dict[h][:n]
        h_preds_trimmed = {m: v[:n] for m, v in h_preds.items()}

        dm_pvalues, dm_stats = ev.run_dm_matrix(h_preds_trimmed, y_true_trimmed, h)
        ev.plot_dm_heatmap(dm_pvalues, h)
        print(f"\nDM p-values (t+{h}):")
        print(dm_pvalues.round(3).to_string())

    # Summary: best model per horizon
    print("\n" + "="*60)
    print("  BEST MODEL SUMMARY (by RMSE)")
    print("="*60)
    for h in HORIZONS:
        rmse_h = df_results.xs(f"t+{h}", level="Horizon")["RMSE"].astype(float)
        best   = rmse_h.idxmin()
        print(f"  t+{h}: {best} (RMSE={rmse_h[best]:.4f})")

    return df_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hanoi Temperature Forecasting Pipeline")
    parser.add_argument("--fast",    action="store_true", help="Skip LSTM models")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Run single horizon only (1, 3, or 7)")
    args = parser.parse_args()

    if args.horizon:
        global HORIZONS
        HORIZONS = [args.horizon]

    _import_all()
    t_start = time.time()

    print("\n" + "="*60)
    print("  Hanoi Temperature Forecasting — Research Pipeline")
    print("  STL Decomposition vs End-to-End Models")
    print("="*60)

    df, train, val, test, df_feat = step_data(args)
    components_train, components_val, seasonal_pattern = step_stl(train, val)
    models = step_train(train, val, df, df_feat, components_train, components_val,
                         seasonal_pattern, fast_mode=args.fast)
    all_preds, target_test = step_forecast(
        models, train, val, test, df, df_feat,
        components_train, seasonal_pattern, fast_mode=args.fast)
    df_results = step_evaluate(all_preds, target_test, fast_mode=args.fast)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
