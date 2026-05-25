# =============================================================================
# 05_evaluation.py — Metrics, Diebold-Mariano test, result tables
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import itertools
import warnings
import os

warnings.filterwarnings("ignore")
from config import *


# =============================================================================
# 1. POINT FORECAST METRICS
# =============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def skill_score(y_true: np.ndarray, y_pred: np.ndarray, naive_pred: np.ndarray) -> float:
    """Skill score vs naive (persistence) baseline. Positive = better than naive."""
    return 1 - rmse(y_true, y_pred) / rmse(y_true, naive_pred)

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          naive_pred: np.ndarray = None) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    metrics = {
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAE":   round(mae(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
    }
    if naive_pred is not None:
        naive_pred = np.array(naive_pred)[mask]
        metrics["Skill"] = round(skill_score(y_true, y_pred, naive_pred), 4)

    return metrics


# =============================================================================
# 2. DIEBOLD-MARIANO TEST
# =============================================================================

def diebold_mariano_test(y_true: np.ndarray, pred_a: np.ndarray,
                          pred_b: np.ndarray, h: int = DM_H,
                          loss: str = DM_LOSS) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[d_t] = 0  (models A and B have equal forecast accuracy)
    H1: E[d_t] ≠ 0  (one model is significantly more accurate)

    Parameters
    ----------
    y_true : actual values
    pred_a : forecasts from model A
    pred_b : forecasts from model B
    h      : forecast horizon (steps ahead) — used for lag correction
    loss   : "squared" or "absolute"

    Returns
    -------
    dict with DM statistic, p-value, and which model wins
    """
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)

    mask = ~(np.isnan(y_true) | np.isnan(pred_a) | np.isnan(pred_b))
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]

    e_a = y_true - pred_a
    e_b = y_true - pred_b

    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    else:
        d = np.abs(e_a) - np.abs(e_b)

    n = len(d)
    d_bar = np.mean(d)

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    # Variance with HAC covariance (Newey-West style, h-1 lags)
    gamma = [np.mean((d - d_bar) * np.roll(d - d_bar, k)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        var_d = np.var(d) / n

    dm_stat = d_bar / np.sqrt(var_d)

    # t-distribution with n-1 df (HLN correction)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    result = {
        "dm_stat":  round(dm_stat, 4),
        "p_value":  round(p_value, 4),
        "n_obs":    n,
        "winner":   "A" if dm_stat > 0 else "B",
        "significant": p_value < 0.05,
    }
    return result


def run_dm_matrix(predictions: dict, y_true: np.ndarray,
                   horizon: int, save_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    Run DM test for all pairs of models.
    Returns a DataFrame with p-values (lower triangle) and DM stats (upper).
    """
    model_names = list(predictions.keys())
    n = len(model_names)
    dm_pvalues  = pd.DataFrame(np.nan, index=model_names, columns=model_names)
    dm_stats    = pd.DataFrame(np.nan, index=model_names, columns=model_names)

    for a, b in itertools.combinations(model_names, 2):
        res = diebold_mariano_test(y_true, predictions[a], predictions[b], h=horizon)
        dm_pvalues.loc[a, b] = res["p_value"]
        dm_pvalues.loc[b, a] = res["p_value"]
        dm_stats.loc[a, b]   = res["dm_stat"]
        dm_stats.loc[b, a]   = -res["dm_stat"]

    return dm_pvalues, dm_stats


# =============================================================================
# 3. COMPILE RESULTS TABLE
# =============================================================================

def compile_results(all_predictions: dict, y_true_dict: dict,
                     naive_dict: dict) -> pd.DataFrame:
    """
    all_predictions: {model_name: {horizon: np.array}}
    y_true_dict:     {horizon: np.array}
    naive_dict:      {horizon: np.array}  (persistence forecast)

    Returns a MultiIndex DataFrame: model × horizon → metrics
    """
    rows = []
    for model_name, horizon_preds in all_predictions.items():
        for h in HORIZONS:
            if h not in horizon_preds:
                continue
            y_hat  = horizon_preds[h]
            y_true = y_true_dict[h]
            naive  = naive_dict[h]
            m      = compute_all_metrics(y_true, y_hat, naive)
            rows.append({
                "Model":   model_name,
                "Horizon": f"t+{h}",
                **m
            })

    df_results = pd.DataFrame(rows).set_index(["Model", "Horizon"])
    return df_results


# =============================================================================
# 4. VISUALISATIONS
# =============================================================================

def plot_forecast_vs_actual(y_true: pd.Series, predictions: dict,
                              horizon: int, save_dir: str = OUTPUT_DIR,
                              last_n: int = PLOT_LAST_N_DAYS):
    """Line plot: actual vs all model forecasts for last N days."""
    os.makedirs(save_dir, exist_ok=True)
    y = y_true.iloc[-last_n:]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(y.index, y.values, color="black", lw=1.5, label="Actual", zorder=10)

    for model_name, horizon_preds in predictions.items():
        if horizon not in horizon_preds:
            continue
        y_hat = horizon_preds[horizon][-last_n:]
        color = PALETTE.get(model_name, "#999999")
        ax.plot(y.index, y_hat, color=color, lw=1.0, alpha=0.8,
                label=model_name, ls="--" if "STL+" in model_name else "-")

    ax.set_title(f"Forecast vs Actual — horizon t+{horizon} (last {last_n} days)", fontsize=12)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_forecast_h{horizon}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_forecast_h{horizon}.png")


def plot_metrics_heatmap(df_results: pd.DataFrame, metric: str = "RMSE",
                          save_dir: str = OUTPUT_DIR):
    """Heatmap: models (rows) × horizons (cols) for a given metric."""
    os.makedirs(save_dir, exist_ok=True)
    pivot = df_results[metric].unstack(level="Horizon")
    # Reorder horizons
    cols = [f"t+{h}" for h in HORIZONS if f"t+{h}" in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot.astype(float),
        annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.4, ax=ax,
        cbar_kws={"label": metric, "shrink": 0.8},
        annot_kws={"size": 11},
    )

    # Highlight minimum in each column (best model per horizon)
    for col_i, col in enumerate(pivot.columns):
        best_row = pivot[col].astype(float).idxmin()
        row_i = list(pivot.index).index(best_row)
        ax.add_patch(plt.Rectangle((col_i, row_i), 1, 1,
                                    fill=False, edgecolor="#2c3e50", lw=2.5))

    ax.set_title(f"{metric} — all models × horizons\n(box = best per horizon)", fontsize=12)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_heatmap_{metric}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_heatmap_{metric}.png")


def plot_dm_heatmap(dm_pvalues: pd.DataFrame, horizon: int,
                     save_dir: str = OUTPUT_DIR):
    """Heatmap of DM test p-values (< 0.05 = significant difference)."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.eye(len(dm_pvalues), dtype=bool)  # mask diagonal
    cmap = sns.diverging_palette(145, 15, as_cmap=True)

    sns.heatmap(
        dm_pvalues.astype(float),
        annot=True, fmt=".3f", cmap="coolwarm_r",
        vmin=0, vmax=0.2, center=0.05,
        linewidths=0.4, mask=mask, ax=ax,
        cbar_kws={"label": "p-value", "shrink": 0.8},
        annot_kws={"size": 9},
    )
    ax.set_title(f"Diebold-Mariano p-values (H0: equal accuracy) — t+{horizon}\n"
                  f"p < 0.05: significant difference (dark = highly significant)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_dm_h{horizon}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_dm_h{horizon}.png")


def plot_error_distribution(y_true: np.ndarray, predictions: dict,
                              horizon: int, save_dir: str = OUTPUT_DIR):
    """Boxplot of forecast errors per model."""
    os.makedirs(save_dir, exist_ok=True)
    error_data = []
    for model_name, horizon_preds in predictions.items():
        if horizon not in horizon_preds:
            continue
        errors = y_true - horizon_preds[horizon]
        for e in errors:
            error_data.append({"Model": model_name, "Error": e})

    df_err = pd.DataFrame(error_data)
    fig, ax = plt.subplots(figsize=(12, 5))
    order = list(predictions.keys())
    palette = {k: PALETTE.get(k, "#999") for k in order}
    sns.boxplot(data=df_err, x="Model", y="Error", order=order,
                palette=palette, width=0.5, fliersize=2, ax=ax)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Forecast error distribution — t+{horizon}", fontsize=12)
    ax.set_ylabel("Error (°C)")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_errors_h{horizon}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_errors_h{horizon}.png")


def plot_horizon_degradation(df_results: pd.DataFrame, metric: str = "RMSE",
                              save_dir: str = OUTPUT_DIR):
    """
    Line plot showing how accuracy degrades as horizon increases.
    Key visualisation for H2 (horizon decay hypothesis).
    """
    os.makedirs(save_dir, exist_ok=True)
    pivot = df_results[metric].unstack(level="Horizon")
    cols  = [f"t+{h}" for h in HORIZONS if f"t+{h}" in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in pivot.index:
        vals  = pivot.loc[model_name].astype(float).values
        color = PALETTE.get(model_name, "#999999")
        ls    = "--" if "STL+" in model_name else "-"
        ax.plot(cols, vals, marker="o", color=color, lw=1.8,
                label=model_name, ls=ls, markersize=6)

    ax.set_title(f"{metric} degradation across forecast horizons\n"
                  f"(dashed = STL-decomposed, solid = end-to-end)", fontsize=12)
    ax.set_ylabel(f"{metric} (°C)")
    ax.set_xlabel("Forecast horizon")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fig_horizon_degradation_{metric}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig_horizon_degradation_{metric}.png")


# =============================================================================
# MAIN — standalone test
# =============================================================================

if __name__ == "__main__":
    print("Evaluation module loaded.")
    print("Functions: compute_all_metrics, diebold_mariano_test, compile_results")
    print("Plots: plot_forecast_vs_actual, plot_metrics_heatmap, plot_dm_heatmap,")
    print("       plot_error_distribution, plot_horizon_degradation")
