# =============================================================================
# 02_stl_decomposition.py — STL decomposition, diagnostics, residual analysis
# This is the methodological core of the research
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
from config import *


# =============================================================================
# 1. FIT STL
# =============================================================================

def fit_stl(series: pd.Series, period: int = STL_PERIOD, robust: bool = STL_ROBUST) -> dict:
    """
    Fit STL decomposition.
    Returns a dict with the fitted object and its three components.

    Parameters
    ----------
    series  : daily temperature series (training set)
    period  : seasonal period in days (365 for annual)
    robust  : use robust Loess iteration to downweight outliers
    """
    stl = STL(series, period=period, robust=robust)
    result = stl.fit()

    components = {
        "trend":    pd.Series(result.trend,    index=series.index, name="trend"),
        "seasonal": pd.Series(result.seasonal, index=series.index, name="seasonal"),
        "residual": pd.Series(result.resid,    index=series.index, name="residual"),
        "stl_obj":  result,
        "stl_fitted": stl,
    }

    # Strength of trend and seasonality (Wang et al. 2006 formula)
    var_resid    = np.var(result.resid)
    var_tr_resid = np.var(result.trend + result.resid)
    var_se_resid = np.var(result.seasonal + result.resid)

    F_trend    = max(0, 1 - var_resid / var_tr_resid)
    F_seasonal = max(0, 1 - var_resid / var_se_resid)

    components["F_trend"]    = F_trend
    components["F_seasonal"] = F_seasonal

    print(f"\n=== STL Decomposition (period={period}, robust={robust}) ===")
    print(f"  Strength of Trend    (F_T): {F_trend:.4f}   [0=no trend, 1=strong]")
    print(f"  Strength of Seasonal (F_S): {F_seasonal:.4f} [0=no season, 1=strong]")

    return components


# =============================================================================
# 2. RESIDUAL DIAGNOSTICS
# =============================================================================

def diagnose_residuals(residuals: pd.Series, save_dir: str = OUTPUT_DIR) -> dict:
    """
    Check whether STL residuals are white noise.
    If they are NOT white noise → ML models can extract further signal (supports H4).
    """
    os.makedirs(save_dir, exist_ok=True)
    resid = residuals.dropna()

    print(f"\n=== Residual Diagnostics ===")

    # Ljung-Box test
    lb = acorr_ljungbox(resid, lags=[5, 10, 20, 50], return_df=True)
    print("\nLjung-Box test on STL residuals:")
    print(lb[["lb_stat", "lb_pvalue"]].to_string())
    has_autocorr = any(lb["lb_pvalue"] < 0.05)
    print(f"\n→ Residuals {'STILL have' if has_autocorr else 'appear to be'} autocorrelation "
          f"(supports H4: {'YES' if has_autocorr else 'NO — ML may not help much'})")

    # ADF on residuals (should be stationary after decomposition)
    adf_stat, adf_p, *_ = adfuller(resid, regression="c", autolag="AIC")
    print(f"\nADF on residuals: stat={adf_stat:.4f}, p={adf_p:.4f} "
          f"→ {'Stationary ✓' if adf_p < 0.05 else 'Still non-stationary !'}")

    # Normality
    _, shapiro_p = stats.shapiro(resid.sample(min(5000, len(resid)), random_state=42))
    _, jb_p = stats.jarque_bera(resid)
    print(f"\nNormality — Shapiro p={shapiro_p:.4f} | Jarque-Bera p={jb_p:.4f}")

    # Residual stats
    print(f"\nResidual stats: mean={resid.mean():.4f}, std={resid.std():.4f}, "
          f"skew={resid.skew():.4f}, kurt={resid.kurtosis():.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("STL Residual Diagnostics", fontsize=13, fontweight="bold")

    axes[0, 0].plot(resid.index, resid.values, color="#4C72B0", lw=0.7, alpha=0.8)
    axes[0, 0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].set_ylabel("Residual (°C)")

    axes[0, 1].hist(resid.values, bins=60, color="#4C72B0", edgecolor="white", lw=0.4)
    axes[0, 1].set_title("Residual distribution")
    axes[0, 1].set_xlabel("Residual (°C)")

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid, lags=60, ax=axes[1, 0], alpha=0.05, zero=False, color="#4C72B0")
    axes[1, 0].set_title("ACF of residuals")

    stats.probplot(resid.values, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot (normality check)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig4_stl_residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: fig4_stl_residuals.png")

    return {"lb_results": lb, "has_autocorr": has_autocorr,
            "adf_p": adf_p, "shapiro_p": shapiro_p}


# =============================================================================
# 3. VISUALISE DECOMPOSITION
# =============================================================================

def plot_decomposition(series: pd.Series, components: dict, save_dir: str = OUTPUT_DIR):
    """Full 4-panel STL decomposition plot."""
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.08)

    panels = [
        (series,                      "Original series",  "#2c3e50"),
        (components["trend"],         "Trend (Loess)",    "#C44E52"),
        (components["seasonal"],      "Seasonal (period=365)", "#55A868"),
        (components["residual"],      "Residual",         "#8172B2"),
    ]

    axes = []
    for i, (data, title, color) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.plot(data.index, data.values, color=color, lw=0.8 if i == 0 else 1.0)
        ax.set_ylabel(title, fontsize=10)
        if i < 3:
            ax.set_xticks([])
        else:
            ax.set_xlabel("")
        if i == 3:
            ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
        axes.append(ax)

    axes[0].set_title(
        f"STL Decomposition — Hanoi daily temperature\n"
        f"Trend strength: {components['F_trend']:.3f} | Seasonal strength: {components['F_seasonal']:.3f}",
        fontsize=12, fontweight="bold", pad=10
    )

    plt.savefig(os.path.join(save_dir, "fig5_stl_decomposition.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig5_stl_decomposition.png")


# =============================================================================
# 4. EXTRACT SEASONAL PATTERN FOR FUTURE REUSE
# =============================================================================

def extract_seasonal_pattern(components: dict) -> pd.Series:
    """
    Extract the average seasonal cycle (365 values, day-of-year indexed).
    Used in Group B models: Forecast = T̂ + S_pattern[doy] + R̂
    """
    seasonal = components["seasonal"]
    doy_pattern = seasonal.groupby(seasonal.index.dayofyear).mean()
    doy_pattern.index.name = "day_of_year"
    print(f"\nSeasonal pattern extracted: {len(doy_pattern)} day-of-year values")
    print(f"  Peak: day {doy_pattern.idxmax()} (+{doy_pattern.max():.2f}°C)")
    print(f"  Trough: day {doy_pattern.idxmin()} ({doy_pattern.min():.2f}°C)")
    return doy_pattern


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from data_prep import load_and_clean, split_data
    from importlib import import_module
    dp = import_module("data_prep")

    print("Loading data...")
    df = dp.load_and_clean()
    train, val, test = dp.split_data(df)

    print("\nFitting STL on training set...")
    components = fit_stl(train[TARGET])

    print("\nPlotting decomposition...")
    plot_decomposition(train[TARGET], components)

    print("\nDiagnosing residuals...")
    diag = diagnose_residuals(components["residual"])

    print("\nExtracting seasonal pattern...")
    seasonal_pattern = extract_seasonal_pattern(components)

    print("\n[02_stl_decomposition.py] Done.")
