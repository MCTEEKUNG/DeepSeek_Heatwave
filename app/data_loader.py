"""Load pre-computed outputs/ files into clean DataFrames for the Streamlit demo."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
ANALYSIS = OUTPUTS / "analysis"

MODEL_LABELS = {
    "logistic": "Logistic",
    "logistic_balanced_cal": "Logistic (bal+cal)",
    "lgbm": "LightGBM (raw)",
    "lgbm_cal": "LightGBM (Platt)",
    "lgbm_balanced_cal": "LightGBM (bal+cal)",
    "balanced_rf_cal": "Balanced RF (cal)",
    "climatology": "Climatology",
    "persistence": "Persistence",
}

FEATURE_GROUPS = {
    "sm1": "Soil", "sm1_mean7": "Soil", "sm1_mean30": "Soil", "sm1_trend": "Soil",
    "sm3": "Soil", "sm3_mean7": "Soil", "sm3_mean30": "Soil", "sm3_trend": "Soil",
    "tmax_rm": "Thermal", "tmax_mean7": "Thermal",
    "in_hw_today": "Thermal", "hot_frac7": "Thermal",
    "mjo_rmm1": "MJO", "mjo_rmm2": "MJO", "mjo_amp": "MJO",
    "mjo_sin": "MJO", "mjo_cos": "MJO",
    "nino34_lag1m": "ENSO",
    "doy_sin": "Seasonal", "doy_cos": "Seasonal",
}

GROUP_COLORS = {
    "Soil": "#2ca02c", "ENSO": "#1f77b4", "MJO": "#9467bd",
    "Thermal": "#d62728", "Seasonal": "#ff7f0e",
}


def load_bss_ci(target: str = "y_rm") -> pd.DataFrame:
    """Bootstrap CI table filtered to target + BSS metric, with friendly model labels."""
    df = pd.read_csv(ANALYSIS / "bootstrap_ci.csv")
    df = df[(df["target"] == target) & (df["metric"] == "bss")].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.reset_index(drop=True)


def load_regime_bss(target: str = "y_rm") -> pd.DataFrame:
    """ENSO regime BSS and base rate filtered to target, with friendly model labels."""
    df = pd.read_csv(ANALYSIS / "regime_by_enso.csv")
    df = df[df["target"] == target].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.reset_index(drop=True)


def load_permutation_importance(target: str = "y_rm", model: str = "lgbm") -> pd.DataFrame:
    """Permutation importance with feature group labels and group colors."""
    df = pd.read_csv(ANALYSIS / "permutation_importance.csv")
    df = df[(df["target"] == target) & (df["model"] == model)].copy()
    df["group"] = df["feature"].map(FEATURE_GROUPS).fillna("Other")
    df["color"] = df["group"].map(GROUP_COLORS).fillna("#7f7f7f")
    return df.reset_index(drop=True)


def load_calibration_decomp(target: str = "y_rm") -> pd.DataFrame:
    """Brier score decomposition filtered to target, with friendly model labels."""
    df = pd.read_csv(ANALYSIS / "calibration_decomp.csv")
    df = df[df["target"] == target].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.reset_index(drop=True)


def load_predictions(target: str = "y_rm") -> pd.DataFrame:
    """Raw per-day predictions for reliability curve computation."""
    df = pd.read_csv(OUTPUTS / "predictions.csv", parse_dates=["date"])
    return df[df["target"] == target].copy().reset_index(drop=True)
