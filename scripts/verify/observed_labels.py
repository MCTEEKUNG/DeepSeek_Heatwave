"""Shared observed-label engine for the verification system.

Assembles per-province ERA5 features AND observed heatwave labels using a
**frozen** p90 threshold (passed in as a parameter — never recomputed here).

This bridges the gap between the two existing modes in build_provinces_features:
  operational=True  -> uses frozen thr90 BUT skips targets (hw_grid=None)
  operational=False -> builds targets BUT recomputes p90 live

build_labeled_frame() does both correctly: frozen thr90 + full-series labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# scripts/ is the module root; observed_labels.py is one level deeper (scripts/verify/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heatwave_target import load_tmax_celsius, hot_days, flag_heatwaves
from build_dataset import (
    lookback_features, mjo_features, nino_lagged_daily,
    weekly_event_targets, LEADS, MIN_RUN, INDICES_DIR,
)
from province_grid import load_provinces, province_series, REGIONS

# ---------------------------------------------------------------------------
# Feature columns — canonical 28-column list mirroring train_provinces.FEATURES_P
# (defined here directly to avoid import-time side-effects from train_provinces.py)
# ---------------------------------------------------------------------------
_LOCAL_FEATURES: list[str] = [
    "sm1", "sm1_mean7", "sm1_mean30", "sm1_trend",
    "sm3", "sm3_mean7", "sm3_mean30", "sm3_trend",
    "tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7",
]
_SHARED_FEATURES: list[str] = [
    "mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos",
    "nino34_lag1m", "doy_sin", "doy_cos",
]
_PROVINCE_STATIC: list[str] = ["lat", "lon"] + [f"region_{r}" for r in REGIONS]

FEATURES_P: list[str] = _LOCAL_FEATURES + _SHARED_FEATURES + _PROVINCE_STATIC

assert len(FEATURES_P) == 28, f"FEATURES_P count mismatch: {len(FEATURES_P)}"


# ---------------------------------------------------------------------------
# Internal helper — mirrors _soil_grid in build_provinces_dataset.py
# ---------------------------------------------------------------------------

def _soil_grid(layer: int, soil_dir: Path) -> xr.DataArray:
    """Load a soil-moisture layer grid from NetCDF files in soil_dir."""
    files = sorted(soil_dir.glob(f"era5_sm_l{layer}_thailand_*.nc"))
    if not files:
        raise FileNotFoundError(f"ไม่พบ soil moisture ชั้น {layer} ใน {soil_dir}")
    ds = xr.open_mfdataset([str(p) for p in files], combine="by_coords")
    da = ds[f"swvl{layer}"].load()
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if "number" in da.coords:
        da = da.drop_vars("number")
    return da.sortby("time")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_labeled_frame(
    tmax_dir: Path,
    soil_dir: Path,
    *,
    frozen_thr90: xr.DataArray,
    leads: list[int] = LEADS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assemble per-province features + observed heatwave labels (frozen thr90).

    Parameters
    ----------
    tmax_dir:
        Directory containing ERA5 Tmax NetCDF files
        (``era5_tmax_thailand_*.nc`` pattern).
    soil_dir:
        Directory containing ERA5 soil-moisture NetCDF files
        (``era5_sm_l{1,3}_thailand_*.nc`` patterns).
    frozen_thr90:
        Pre-computed p90 climatology grid — dims (dayofyear, lat, lon) as
        produced by ``doy_window_percentile`` / ``_load_frozen_climatology()``.
        NEVER recomputed inside this function.
    leads:
        Lead times in weeks to compute targets for (default: [2, 3, 4, 5, 6]).
    verbose:
        Print progress messages when True.

    Returns
    -------
    pd.DataFrame
        Columns: FEATURES_P (28) + ["province_id", "date",
        "y_rm_l2", "y_rm_l3", "y_rm_l4", "y_rm_l5", "y_rm_l6"].
        y_rm_l{L} is 1.0 / 0.0 / NaN (NaN = window not yet closed).
    """
    def log(m: str) -> None:
        if verbose:
            print(m, flush=True)

    # ------------------------------------------------------------------
    # Step 1: Load Tmax + apply FROZEN threshold (rule #1: never recompute)
    # ------------------------------------------------------------------
    log("[verify] โหลด Tmax grid + ใช้เกณฑ์ p90 แช่แข็ง ...")
    t_grid = load_tmax_celsius(sorted(tmax_dir.glob("era5_tmax_thailand_*.nc")))
    hot_grid = hot_days(t_grid, frozen_thr90)          # frozen thr90 — NOT recomputed
    hw_grid = flag_heatwaves(hot_grid, min_len=MIN_RUN)  # full-series streak detection

    # ------------------------------------------------------------------
    # Step 2: Load soil moisture grids
    # ------------------------------------------------------------------
    log("[verify] โหลด soil moisture grid ชั้น 1, 3 ...")
    sm1_grid = _soil_grid(1, soil_dir)
    sm3_grid = _soil_grid(3, soil_dir)

    # ------------------------------------------------------------------
    # Step 3: Load shared climate indices (MJO + Niño3.4)
    # ------------------------------------------------------------------
    log("[verify] โหลด MJO + Niño3.4 indices ...")
    mjo = mjo_features(INDICES_DIR / "mjo_rmm.csv")

    # ------------------------------------------------------------------
    # Step 4: For each province, build features + labels then merge
    # ------------------------------------------------------------------
    pv = load_provinces()
    frames: list[pd.DataFrame] = []

    for r in pv.itertuples():
        # --- features ---
        daily = pd.DataFrame({
            "sm1": province_series(sm1_grid, r.lat, r.lon),
            "sm3": province_series(sm3_grid, r.lat, r.lon),
            "tmax_rm": province_series(t_grid, r.lat, r.lon),
            "hot_rm": province_series(hot_grid.astype(float), r.lat, r.lon),
        }).sort_index()

        feat = lookback_features(daily)
        feat = feat.join(mjo, how="left")
        feat["nino34_lag1m"] = nino_lagged_daily(INDICES_DIR / "nino34.csv", feat.index)
        feat["lat"] = float(r.lat)
        feat["lon"] = float(r.lon)
        for reg in REGIONS:
            feat[f"region_{reg}"] = 1.0 if r.region == reg else 0.0
        feat["province_id"] = int(r.id)
        feat["date"] = feat.index

        # --- observed labels (hw_grid already has full-series streak detection) ---
        hw_rm = province_series(hw_grid.astype(float), r.lat, r.lon).sort_index()
        tg = weekly_event_targets(hw_rm, leads)           # NaN for incomplete windows
        tg = tg.rename(columns={f"lead{L}": f"y_rm_l{L}" for L in leads})
        tg["province_id"] = int(r.id)
        tg["date"] = tg.index

        feat_r = feat.reset_index(drop=True)
        tg_r = tg.reset_index(drop=True)
        frames.append(feat_r.merge(tg_r.drop(columns=["province_id"]), on="date", how="left"))

    log(f"[verify] รวม {len(pv)} จังหวัด เสร็จแล้ว")
    return pd.concat(frames, ignore_index=True)
