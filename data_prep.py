# =============================================================================
# 01_data_prep.py — Data loading, cleaning, EDA, stationarity tests
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
from config import *


# =============================================================================
# 1. LOAD & CLEAN
# =============================================================================

def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load Hanoi weather CSV, parse dates, set index, clean missing values.
    Returns a daily-frequency DataFrame with temp as primary target.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    df.index.freq = pd.infer_freq(df.index)    # try to infer 'D'

    # Keep columns useful for feature engineering
    keep_cols = [
        "tempmax", "tempmin", "temp", "feelslike",
        "humidity", "precip", "precipprob",
        "windspeed", "windgust", "winddir",
        "sealevelpressure", "cloudcover",
        "visibility", "solarradiation", "uvindex",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Drop data beyond test set (2025 không dùng đến)
    df = df.loc[:"2024-12-31"]

    # Report missing values
    missing = df.isnull().sum()
    print("=== Missing values ===")
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "None — dataset is complete.")

    # Interpolate small gaps in target (≤ 3 consecutive days)
    df[TARGET] = df[TARGET].interpolate(method="time", limit=3)

    # Forward-fill remaining exogenous features
    df = df.ffill().bfill()

    print(f"\nDataset: {df.index[0].date()} → {df.index[-1].date()} | {len(df):,} observations")
    return df


def data_quality_report(df: pd.DataFrame):
    """
    Comprehensive data quality assessment.
    """

    print("="*80)
    print("DATA QUALITY SUMMARY")
    print("="*80)

    # 1. Basic information
    print(f"\n{'='*80}")
    print("1. DATASET DIMENSIONS")
    print(f"{'='*80}")

    print(f"Total Records: {len(df):,}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Time Span: {(df.index.max() - df.index.min()).days} days")

    # 2. Duplicate records
    print(f"\n{'='*80}")
    print("2. DUPLICATE RECORDS")
    print(f"{'='*80}")

    n_duplicates = df.duplicated().sum()

    print(
        f"Duplicate Rows: "
        f"{n_duplicates} "
        f"({n_duplicates/len(df)*100:.2f}%)"
    )

    # 3. Missing values summary
    print(f"\n{'='*80}")
    print("3. MISSING VALUES SUMMARY")
    print(f"{'='*80}")

    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]

    print(
        f"Total Missing Values: "
        f"{total_missing:,} "
        f"({total_missing/total_cells*100:.2f}% of all cells)"
    )

    print(
        f"Features with Missing Data: "
        f"{(df.isnull().sum() > 0).sum()}"
    )

    print(
        f"Features without Missing Data: "
        f"{(df.isnull().sum() == 0).sum()}"
    )

    # 4. Data type distribution
    print(f"\n{'='*80}")
    print("4. DATA TYPE DISTRIBUTION")
    print(f"{'='*80}")

    dtype_counts = df.dtypes.value_counts()

    for dtype, count in dtype_counts.items():
        print(f"{dtype}: {count} features")

    # 5. Value range validation
    print(f"\n{'='*80}")
    print("5. VALUE RANGE VALIDATION")
    print(f"{'='*80}")

    validation_checks = []

    humidity_invalid = (
        (df['humidity'] < 0)
        | (df['humidity'] > 100)
    ).sum()

    validation_checks.append(
        ('Humidity', '0-100%', humidity_invalid,
         'Valid' if humidity_invalid == 0 else 'WARNING')
    )

    cloudcover_invalid = (
        (df['cloudcover'] < 0)
        | (df['cloudcover'] > 100)
    ).sum()

    validation_checks.append(
        ('Cloud Cover', '0-100%', cloudcover_invalid,
         'Valid' if cloudcover_invalid == 0 else 'WARNING')
    )

    precip_invalid = (df['precip'] < 0).sum()

    validation_checks.append(
        ('Precipitation', '>= 0 mm', precip_invalid,
         'Valid' if precip_invalid == 0 else 'WARNING')
    )

    windspeed_invalid = (df['windspeed'] < 0).sum()

    validation_checks.append(
        ('Wind Speed', '>= 0 km/h', windspeed_invalid,
         'Valid' if windspeed_invalid == 0 else 'WARNING')
    )

    temp_invalid = (
        (df['temp'] < -5)
        | (df['temp'] > 50)
    ).sum()

    validation_checks.append(
        ('Temperature', '-5 to 50°C', temp_invalid,
         'Valid' if temp_invalid == 0 else 'WARNING')
    )

    for check in validation_checks:
        status_symbol = '✓' if check[3] == 'Valid' else '⚠'

        print(
            f"{status_symbol} {check[0]}: "
            f"Expected {check[1]} | "
            f"Invalid: {check[2]} | "
            f"Status: {check[3]}"
        )

    # Overall score
    print(f"\n{'='*80}")
    print("7. OVERALL DATA QUALITY ASSESSMENT")
    print(f"{'='*80}")

    quality_score = 100 - (total_missing / total_cells * 100)

    print(f"Data Completeness Score: {quality_score:.2f}%")

    status = (
        "EXCELLENT"
        if quality_score > 95
        else "GOOD"
        if quality_score > 90
        else "FAIR"
    )

    print(f"Data Quality Status: {status}")

    print(f"\n{'Dataset is ready for preprocessing and feature engineering!' if quality_score > 90 else 'Consider addressing data quality issues before proceeding.'}")

    return {
        "quality_score": quality_score,
        "duplicates": n_duplicates,
        "missing_values": total_missing,
    }


# =============================================================================
# 2. TRAIN / VAL / TEST SPLIT
# =============================================================================

def split_data(df: pd.DataFrame):
    """Chronological split — never random for time series."""
    train = df.loc[:TRAIN_END]
    val   = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1) : VAL_END]
    test  = df.loc[pd.Timestamp(VAL_END) + pd.Timedelta(days=1):]
    print(f"\nTrain : {train.index[0].date()} → {train.index[-1].date()} ({len(train):,} obs)")
    print(f"Val   : {val.index[0].date()}   → {val.index[-1].date()}   ({len(val):,} obs)")
    print(f"Test  : {test.index[0].date()}  → {test.index[-1].date()}  ({len(test):,} obs)")
    return train, val, test


# =============================================================================
# 3. STATIONARITY TESTS
# =============================================================================

def run_stationarity_tests(series: pd.Series, name: str = "temp") -> dict:
    """
    ADF test (H0: unit root = non-stationary) and
    KPSS test (H0: trend-stationary).
    Report both and draw conclusions.
    """
    print(f"\n{'='*55}")
    print(f"  Stationarity tests — {name}")
    print(f"{'='*55}")

    # ADF
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(series.dropna(), regression="ct", autolag="AIC")
    print(f"\nADF Test (H0: unit root / non-stationary)")
    print(f"  Statistic : {adf_stat:.4f}")
    print(f"  p-value   : {adf_p:.4f}  → {'Reject H0 (stationary)' if adf_p < 0.05 else 'Fail to reject H0 (non-stationary)'}")
    print(f"  Lags used : {adf_lags}")
    for k, v in adf_crit.items():
        print(f"  Critical {k}: {v:.4f}")

    # KPSS
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series.dropna(), regression="ct", nlags="auto")
    print(f"\nKPSS Test (H0: trend-stationary)")
    print(f"  Statistic : {kpss_stat:.4f}")
    print(f"  p-value   : {kpss_p:.4f}  → {'Reject H0 (non-stationary)' if kpss_p < 0.05 else 'Fail to reject H0 (stationary)'}")

    # Ljung-Box on raw series
    lb = acorr_ljungbox(series.dropna(), lags=[10, 20, 50], return_df=True)
    print(f"\nLjung-Box autocorrelation test:")
    print(lb[["lb_stat", "lb_pvalue"]].to_string())

    return {
        "adf_stat": adf_stat, "adf_p": adf_p,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p,
    }


# =============================================================================
# 4. EDA PLOTS
# =============================================================================

def plot_eda(df: pd.DataFrame, save_dir: str = OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    series = df[TARGET]

    # --- Figure 1: Time series + annual average trend ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    axes[0].plot(series.index, series.values, color="#4C72B0", lw=0.7, alpha=0.8)
    axes[0].set_title("Hanoi daily mean temperature (2015–2025)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Temperature (°C)")
    annual = series.resample("YE").mean()
    axes[0].plot(annual.index, annual.values, color="#C44E52", lw=2.5, label="Annual mean", zorder=5)
    axes[0].legend()

    # Monthly boxplot
    df_copy = df.copy()
    df_copy["month"] = df_copy.index.month
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    sns.boxplot(data=df_copy, x="month", y=TARGET, ax=axes[1],
                palette="coolwarm", width=0.6, fliersize=2)
    axes[1].set_xticklabels(month_names)
    axes[1].set_title("Temperature distribution by month — seasonal pattern", fontsize=12)
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_timeseries_seasonality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig1_timeseries_seasonality.png")

    # --- Figure 2: ACF / PACF ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=400, ax=axes[0], alpha=0.05, zero=False, color="#4C72B0")
    axes[0].set_title("ACF — daily temperature (lags 1–400)", fontsize=12)
    plot_pacf(series.dropna(), lags=50, ax=axes[1], alpha=0.05, zero=False, method="ywm", color="#4C72B0")
    axes[1].set_title("PACF — daily temperature (lags 1–50)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_acf_pacf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig2_acf_pacf.png")

    # --- Figure 3: Correlation heatmap of features ---
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("Feature correlation matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fig3_correlation.png")


# =============================================================================
# 5. FEATURE ENGINEERING (for ML/DL models)
# =============================================================================

def make_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Create lag features, rolling statistics, and calendar features.
    These are used by XGBoost and LSTM (end-to-end, Group A).
    """
    df = df.copy()
    s = df[target]

    # Lag features
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = s.shift(lag)

    # Rolling statistics
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = s.shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"]  = s.shift(1).rolling(w).std()
        df[f"rolling_min_{w}"]  = s.shift(1).rolling(w).min()
        df[f"rolling_max_{w}"]  = s.shift(1).rolling(w).max()

    # Calendar features (sin/cos encoding for cyclical nature)
    df["doy"]       = df.index.dayofyear
    df["month"]     = df.index.month
    df["week"]      = df.index.isocalendar().week.astype(int)
    df["doy_sin"]   = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["doy"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]  = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week"] / 52)

    return df.dropna()


# =============================================================================
# MAIN — run standalone for inspection
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_clean()

    print("\nAssessing data quality...")
    data_quality_report(df)

    train, val, test = split_data(df)

    print("\nRunning stationarity tests on training set...")
    stats_results = run_stationarity_tests(train[TARGET], name="temp (train)")

    print("\nGenerating EDA plots...")
    plot_eda(df)

    print("\nBuilding feature matrix...")
    df_feat = make_features(df)
    feature_cols = [c for c in df_feat.columns if c != TARGET]
    print(f"Features created: {len(feature_cols)} columns")
    print(df_feat[feature_cols[:5]].head(3).to_string())

    print("\n[01_data_prep.py] Done.")
