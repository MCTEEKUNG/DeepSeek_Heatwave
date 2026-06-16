"""ประกอบ pooled dataset รายจังหวัด (77) สำหรับโมเดล sub-seasonal per-province.

reuse: heatwave_target (per-cell percentile/heatwave), build_dataset.lookback_features
+ mjo_features + nino_lagged_daily + weekly_event_targets. 1 แถว = (จังหวัด x วันออกพยากรณ์).
ตัวสร้าง feature (build_provinces_features) ใช้ทั้ง train (build) และ serve (predict_provinces) -> parity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatwave_target import load_tmax_celsius, doy_window_percentile, hot_days, flag_heatwaves
from build_dataset import (lookback_features, mjo_features, nino_lagged_daily,
                           weekly_event_targets, LEADS, MIN_RUN, PCTL_WINDOW,
                           INDICES_DIR, TMAX_DIR, SOIL_DIR)
from province_grid import load_provinces, province_series, REGIONS

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "data" / "processed" / "dataset_provinces.parquet"
RECENT_TMAX_DIR = TMAX_DIR.parent.parent / "raw_recent" / "tmax_thailand"
RECENT_SOIL_DIR = SOIL_DIR.parent.parent / "raw_recent" / "soil_moisture_thailand"
CLIM_PROV_FILE = ROOT / "models" / "climatology_provinces.pkl"


def _load_frozen_climatology() -> dict:
    import pickle
    # pickle ปลอดภัยที่นี่: ไฟล์สร้างเองด้วย freeze_provinces_climatology.py
    # บนเครื่อง/CI เดียวกัน ไม่ได้โหลดจากแหล่งภายนอกที่ไม่น่าเชื่อถือ
    if not CLIM_PROV_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบ {CLIM_PROV_FILE.name} — รัน `python scripts/freeze_provinces_climatology.py` ก่อน"
        )
    with open(CLIM_PROV_FILE, "rb") as fh:
        return pickle.load(fh)


LOCAL_FEATURES = ["sm1", "sm1_mean7", "sm1_mean30", "sm1_trend",
                  "sm3", "sm3_mean7", "sm3_mean30", "sm3_trend",
                  "tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7"]
SHARED_FEATURES = ["mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos",
                   "nino34_lag1m", "doy_sin", "doy_cos"]


def province_static_columns() -> list[str]:
    return ["lat", "lon"] + [f"region_{r}" for r in REGIONS]


def _soil_grid(layer: int, soil_dir: Path = SOIL_DIR) -> xr.DataArray:
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


def build_provinces_features(verbose: bool = True, operational: bool = False) -> tuple[pd.DataFrame, "xr.DataArray | None"]:
    """คืน (feat_all, hw_grid). feat_all = pooled features ทุกจังหวัด (ยังไม่มี target).

    reuse ตัวเดียวทั้ง train และ serve เพื่อ parity. hw_grid = heatwave per-cell (ไว้ทำ target ใน build()).
    operational=True: ใช้ข้อมูล raw_recent + thr90 แช่แข็ง + MJO impute; hw_grid คืน None.
    """
    def log(m):
        if verbose:
            print(m, flush=True)
    tmax_dir = RECENT_TMAX_DIR if operational else TMAX_DIR
    soil_dir = RECENT_SOIL_DIR if operational else SOIL_DIR
    log("[prov] โหลด Tmax grid + เกณฑ์ p90 ราย doy ราย cell ...")
    t_grid = load_tmax_celsius(sorted(tmax_dir.glob("era5_tmax_thailand_*.nc")))
    if operational:
        thr90 = _load_frozen_climatology()["thr90_grid"]
    else:
        thr90 = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)
    hot_grid = hot_days(t_grid, thr90)
    hw_grid = None if operational else flag_heatwaves(hot_grid, min_len=MIN_RUN)
    log("[prov] โหลด soil moisture grid ชั้น 1, 3 ...")
    sm1_grid, sm3_grid = _soil_grid(1, soil_dir), _soil_grid(3, soil_dir)
    mjo = mjo_features(INDICES_DIR / "mjo_rmm.csv")
    pv = load_provinces()

    frames = []
    for r in pv.itertuples():
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
        frames.append(feat.reset_index(drop=True))
    feat_all = pd.concat(frames, ignore_index=True)
    log(f"[prov] features: {len(feat_all)} แถว x {len(pv)} จังหวัด")
    if operational:
        from predict import impute_neutral_mjo, load_climatology
        means = (load_climatology() or {}).get("mjo_means")
        tmp = feat_all.set_index("date")
        tmp, imputed = impute_neutral_mjo(tmp, means)
        feat_all = tmp.reset_index()
        feat_all.attrs["mjo_imputed_dates"] = imputed
    return feat_all, hw_grid


def build(verbose: bool = True) -> pd.DataFrame:
    feat_all, hw_grid = build_provinces_features(verbose=verbose)
    pv = load_provinces()
    tgt_frames = []
    for r in pv.itertuples():
        hw_s = province_series(hw_grid.astype(float), r.lat, r.lon).sort_index()
        tg = weekly_event_targets(hw_s, LEADS)
        tg = tg.rename(columns={f"lead{L}": f"y_rm_l{L}" for L in LEADS})
        tg["province_id"] = int(r.id)
        tg["date"] = tg.index
        tgt_frames.append(tg.reset_index(drop=True))
    targets = pd.concat(tgt_frames, ignore_index=True)
    df = feat_all.merge(targets, on=["province_id", "date"], how="left")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    if verbose:
        print(f"[OK] {OUT_FILE} rows={len(df)}")
        for L in LEADS:
            m = df[f"y_rm_l{L}"].notna()
            print(f"     lead {L}: ใช้ได้ {int(m.sum())} แถว | base_rate={df.loc[m, f'y_rm_l{L}'].mean():.3f}")
    return df


def _selftest() -> None:
    """leakage: เปลี่ยน Tmax 'วันอนาคต' ของ cell หนึ่ง -> feature local อดีตต้องไม่ขยับ (reuse lookback_features)."""
    idx = pd.date_range("2020-01-01", periods=60, freq="D")
    daily = pd.DataFrame({"sm1": np.linspace(0.3, 0.4, 60), "sm3": np.linspace(0.35, 0.45, 60),
                          "tmax_rm": np.linspace(30, 38, 60), "hot_rm": 0.0}, index=idx)
    fa = lookback_features(daily)
    d2 = daily.copy(); d2.iloc[-1, d2.columns.get_loc("tmax_rm")] = 99.0
    fb = lookback_features(d2)
    same = fa.iloc[:-1].drop(columns=["doy"]).fillna(-1) == fb.iloc[:-1].drop(columns=["doy"]).fillna(-1)
    assert same.all().all(), "feature local leakage: อนาคตกระทบอดีต"
    pv = load_provinces()
    cols = province_static_columns()
    assert cols == ["lat", "lon"] + [f"region_{r}" for r in REGIONS], cols
    print(f"[OK] leakage-safe local features + province_static {len(cols)} คอลัมน์")
    print("[OK] self-test ผ่าน")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        build()
