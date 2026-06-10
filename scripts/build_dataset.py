"""
ประกอบ "ตารางข้อมูล" สำหรับเทรนโมเดล: 1 แถว = 1 วันออกพยากรณ์ (issue date)
คอลัมน์ = features ที่รู้ได้ ณ วันนั้น + target ของหน้าต่างเป้าหมายที่ lead 2-6 สัปดาห์

นิยาม target เชิงพื้นที่ (ตัดสินใจ 2026-06-11):
  หลัก   regional-mean : เฉลี่ย Tmax ทั้งกรอบไทย (ถ่วงน้ำหนัก cos(lat)) ก่อน
         แล้วค่อยหาเกณฑ์ p90 ตามวันในปี (±15 วัน) + ติดต่อกัน >= 3 วัน
         เหตุผล: สัญญาณ sub-seasonal มาจากตัวขับเคลื่อนระดับใหญ่
         (soil moisture / ENSO / MJO) ซึ่งสอดคล้องกับเหตุการณ์ "ระดับภูมิภาค"
         และ series เดียวตีความ/วัดผลง่าย เป็นนิยามตั้งต้นที่ปลอดภัยที่สุด
  รอง    area-fraction (ablation) : หา heatwave รายเซลล์ (เกณฑ์ p90 ราย doy ต่อเซลล์
         + >=3 วันติดต่อกันต่อเซลล์) แล้ววันไหนสัดส่วนพื้นที่ที่เป็น heatwave
         >= AF_THRESHOLD (0.15) ถือเป็น "วัน heatwave เชิงพื้นที่"
         จับเหตุการณ์เฉพาะถิ่นที่ค่าเฉลี่ยภูมิภาคกลบหาย
  เสริม  regional-mean ที่ p95 (ตาม spec ให้เทียบ 90 vs 95)

target รายสัปดาห์: y = 1 ถ้ามีวัน heatwave อย่างน้อย 1 วันในหน้าต่าง 7 วัน
ที่เริ่ม L สัปดาห์หลังวันออกพยากรณ์ (L = 2..6) ; หน้าต่างต้องมีข้อมูลครบ 7 วัน
(ข้อมูลมีเฉพาะ ม.ค.-ก.ค. — หน้าต่างที่ทะลุออกนอกช่วงจะเป็น NaN และถูกตัดตอนเทรน)

Features (ทั้งหมด "รู้ได้ ณ วันออกพยากรณ์" — มองย้อนหลังเท่านั้น กัน leakage):
  - soil moisture ชั้น 1/3 เฉลี่ยภูมิภาค: ค่าวันนี้, ค่าเฉลี่ย 7/30 วันย้อนหลัง, trend (7-30)
  - Tmax ภูมิภาค: ค่าวันนี้, เฉลี่ย 7 วันย้อนหลัง
  - สถานะปัจจุบัน: in_hw_today (อยู่ในช่วง heatwave ไหม), hot_frac7 (สัดส่วนวันร้อนใน 7 วันหลัง)
  - MJO: RMM1, RMM2, amplitude, phase เข้ารหัสเป็น sin/cos
  - Niño3.4 anomaly "ของเดือนก่อนหน้า" (lag 1 เดือน — เดือนปัจจุบันยังไม่ประกาศจริง)
  - ฤดูกาล: doy_sin, doy_cos

ใช้งาน:  python build_dataset.py        # สร้าง data/processed/dataset.csv
         python build_dataset.py test   # รันเฉพาะ self-test (ไม่แตะข้อมูลจริง)
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heatwave_target import (
    load_tmax_celsius,
    doy_window_percentile,
    hot_days,
    flag_heatwaves,
)

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TMAX_DIR = RAW / "tmax_thailand"
SOIL_DIR = RAW / "soil_moisture_thailand"
INDICES_DIR = PROCESSED / "indices"
OUT_FILE = PROCESSED / "dataset.csv"

LEADS = [2, 3, 4, 5, 6]          # สัปดาห์
TARGET_WINDOW_DAYS = 7
MIN_RUN = 3                      # วันติดต่อกันขั้นต่ำของ heatwave
AF_THRESHOLD = 0.15              # สัดส่วนพื้นที่ขั้นต่ำของนิยาม area-fraction
PCTL_WINDOW = 15                 # +/- วันรอบ doy สำหรับเกณฑ์ percentile


# ---------------------------------------------------------------- utilities

def _spatial_dims(da: xr.DataArray) -> list[str]:
    return [d for d in da.dims if d != "time"]


def _lat_name(da: xr.DataArray) -> str:
    for cand in ("latitude", "lat"):
        if cand in da.dims:
            return cand
    raise KeyError(f"หา dim ละติจูดไม่เจอใน {list(da.dims)}")


def regional_mean(da: xr.DataArray) -> xr.DataArray:
    """เฉลี่ยเชิงพื้นที่ถ่วงน้ำหนัก cos(lat) — เซลล์ใกล้ศูนย์สูตรพื้นที่จริงใหญ่กว่า."""
    lat = _lat_name(da)
    w = np.cos(np.deg2rad(da[lat]))
    return da.weighted(w).mean(dim=_spatial_dims(da))


def load_soil_regional(layer: int) -> pd.Series:
    """โหลด soil moisture ชั้นที่กำหนด -> อนุกรมรายวันเฉลี่ยภูมิภาค (m3/m3)."""
    files = sorted(SOIL_DIR.glob(f"era5_sm_l{layer}_thailand_*.nc"))
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ soil moisture ชั้น {layer} ใน {SOIL_DIR}")
    ds = xr.open_mfdataset([str(p) for p in files], combine="by_coords")
    da = ds[f"swvl{layer}"].load()
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if "number" in da.coords:
        da = da.drop_vars("number")
    lo, hi = float(da.min()), float(da.max())
    if lo < -0.05 or hi > 1.0:
        raise ValueError(f"swvl{layer} ค่าหลุดช่วง m3/m3: min={lo}, max={hi}")
    rm = regional_mean(da.sortby("time"))
    return pd.Series(rm.values, index=pd.DatetimeIndex(rm.time.values).normalize(),
                     name=f"sm{layer}")


def to_series(da: xr.DataArray, name: str) -> pd.Series:
    return pd.Series(np.asarray(da.values, dtype=float),
                     index=pd.DatetimeIndex(da.time.values).normalize(), name=name)


# ------------------------------------------------------------ target logic

def weekly_event_targets(daily_flag: pd.Series, leads: list[int]) -> pd.DataFrame:
    """target รายสัปดาห์ต่อ lead: y(t) = max ของ flag ในหน้าต่าง [t+7L, t+7L+6].

    daily_flag: 0/1 บน index วันที่ (เฉพาะวันที่มีข้อมูล)
    คืน DataFrame index เดียวกับ daily_flag ; หน้าต่างที่ข้อมูลไม่ครบ 7 วัน -> NaN
    """
    full = pd.date_range(daily_flag.index.min(), daily_flag.index.max(), freq="D")
    s = daily_flag.astype(float).reindex(full)  # วันไม่มีข้อมูล -> NaN
    out = {}
    for L in leads:
        # rolling(7).max() ที่ตำแหน่ง t+7L+6 ครอบ [t+7L, t+7L+6] พอดี
        # min_periods=7 -> ถ้ามี NaN ในหน้าต่าง ผลเป็น NaN (= หน้าต่างไม่ครบ ตัดทิ้ง)
        rolled = s.rolling(TARGET_WINDOW_DAYS, min_periods=TARGET_WINDOW_DAYS).max()
        out[f"lead{L}"] = rolled.shift(-(7 * L + TARGET_WINDOW_DAYS - 1))
    return pd.DataFrame(out, index=full).loc[daily_flag.index]


# ------------------------------------------------------------ feature logic

def lookback_features(daily: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ feature มองย้อนหลังบนปฏิทินเต็ม (ช่องว่างข้ามปีให้ผลเป็น NaN — ตัดตอนเทรน)."""
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    d = daily.reindex(full)
    f = pd.DataFrame(index=full)

    for c in ("sm1", "sm3"):
        f[c] = d[c]
        f[f"{c}_mean7"] = d[c].rolling(7, min_periods=7).mean()
        f[f"{c}_mean30"] = d[c].rolling(30, min_periods=30).mean()
        f[f"{c}_trend"] = f[f"{c}_mean7"] - f[f"{c}_mean30"]

    f["tmax_rm"] = d["tmax_rm"]
    f["tmax_mean7"] = d["tmax_rm"].rolling(7, min_periods=7).mean()

    f["in_hw_today"] = d["hw_rm"]
    f["hot_frac7"] = d["hot_rm"].rolling(7, min_periods=7).mean()

    doy = pd.Series(full.dayofyear, index=full, dtype=float)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["doy"] = doy  # ไม่ใช่ feature โมเดล — ใช้ทำ baseline climatology
    return f.loc[daily.index]


def mjo_features(path: Path) -> pd.DataFrame:
    mjo = pd.read_csv(path, parse_dates=["date"], index_col="date")
    out = pd.DataFrame(index=mjo.index)
    out["mjo_rmm1"] = mjo["mjo_rmm1"]
    out["mjo_rmm2"] = mjo["mjo_rmm2"]
    out["mjo_amp"] = mjo["mjo_amplitude"]
    ang = (mjo["mjo_phase"] - 1) / 8.0 * 2 * np.pi
    out["mjo_sin"] = np.sin(ang)
    out["mjo_cos"] = np.cos(ang)
    return out


def nino_lagged_daily(path: Path, dates: pd.DatetimeIndex) -> pd.Series:
    """Niño3.4 รายเดือน -> รายวันด้วยค่า 'เดือนก่อนหน้า' (lag 1 เดือน กันใช้ข้อมูลยังไม่ประกาศ)."""
    nino = pd.read_csv(path, parse_dates=["date"], index_col="date")["nino34_anom"]
    lagged = nino.shift(1)  # index รายเดือน (วันที่ 1) ต่อเนื่อง -> เลื่อน 1 แถว = เดือนก่อน
    month_key = pd.DatetimeIndex(dates).to_period("M").to_timestamp()
    return pd.Series(lagged.reindex(month_key).values, index=dates, name="nino34_lag1m")


# ------------------------------------------------------------------- build

def build(verbose: bool = True) -> pd.DataFrame:
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log("=== build_dataset: ประกอบตารางข้อมูล ===")

    # --- 1) Tmax + target หลัก (regional-mean) ---
    files = sorted(TMAX_DIR.glob("era5_tmax_thailand_*.nc"))
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ Tmax ใน {TMAX_DIR}")
    log(f"[1/5] โหลด Tmax {len(files)} ไฟล์ ...")
    t_grid = load_tmax_celsius(files)

    t_rm = regional_mean(t_grid)
    t_rm.attrs["units"] = "degC"
    log("      เกณฑ์ p90/p95 ราย doy (regional-mean) ...")
    thr90_rm = doy_window_percentile(t_rm, q=90, window=PCTL_WINDOW)
    thr95_rm = doy_window_percentile(t_rm, q=95, window=PCTL_WINDOW)
    hot90_rm = hot_days(t_rm, thr90_rm)
    hot95_rm = hot_days(t_rm, thr95_rm)
    hw90_rm = flag_heatwaves(hot90_rm, min_len=MIN_RUN)
    hw95_rm = flag_heatwaves(hot95_rm, min_len=MIN_RUN)

    # --- 2) target รอง (area-fraction รายเซลล์) ---
    log("[2/5] เกณฑ์ p90 ราย doy ต่อเซลล์ (ช้าสุดในไฟล์นี้ ~นาที) ...")
    thr90_cell = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)
    hw_cell = flag_heatwaves(hot_days(t_grid, thr90_cell), min_len=MIN_RUN)
    area_frac = hw_cell.mean(dim=_spatial_dims(hw_cell))
    hw_af = (area_frac >= AF_THRESHOLD)

    # --- 3) soil moisture ---
    log("[3/5] โหลด soil moisture ชั้น 1, 3 ...")
    sm1 = load_soil_regional(1)
    sm3 = load_soil_regional(3)

    daily = pd.DataFrame({
        "tmax_rm": to_series(t_rm, "tmax_rm"),
        "hot_rm": to_series(hot90_rm.astype(float), "hot_rm"),
        "hw_rm": to_series(hw90_rm.astype(float), "hw_rm"),
        "hw_rm95": to_series(hw95_rm.astype(float), "hw_rm95"),
        "area_frac": to_series(area_frac, "area_frac"),
        "hw_af": to_series(hw_af.astype(float), "hw_af"),
    })
    daily = daily.join(sm1, how="left").join(sm3, how="left")
    n_miss_sm = int(daily[["sm1", "sm3"]].isna().sum().sum())
    if n_miss_sm:
        raise ValueError(f"soil moisture ไม่ครบ {n_miss_sm} ค่า — วันที่ Tmax กับ SM ไม่ตรงกัน")

    # --- 4) features ---
    log("[4/5] ประกอบ features (lookback + ดัชนี) ...")
    feat = lookback_features(daily)
    feat = feat.join(mjo_features(INDICES_DIR / "mjo_rmm.csv"), how="left")
    feat["nino34_lag1m"] = nino_lagged_daily(INDICES_DIR / "nino34.csv", feat.index)

    # --- 5) targets ---
    log("[5/5] targets ราย lead ...")
    targets = {}
    for name, flag in (("y_rm", daily["hw_rm"]), ("y_rm95", daily["hw_rm95"]),
                       ("y_af", daily["hw_af"])):
        t = weekly_event_targets(flag, LEADS)
        for L in LEADS:
            targets[f"{name}_l{L}"] = t[f"lead{L}"]
    df = feat.join(pd.DataFrame(targets, index=feat.index))
    df.index.name = "date"

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE)

    log(f"\n[OK] {OUT_FILE}")
    log(f"     rows={len(df)}, features={len(feat.columns) - 1} (ไม่นับ doy), "
        f"targets={len(targets)}")
    for L in LEADS:
        m = df[f"y_rm_l{L}"].notna()
        log(f"     lead {L} สัปดาห์: ใช้ได้ {int(m.sum()):5d} วัน | "
            f"base rate y_rm={df.loc[m, f'y_rm_l{L}'].mean():.3f} "
            f"y_af={df.loc[df[f'y_af_l{L}'].notna(), f'y_af_l{L}'].mean():.3f}")
    return df


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # 1) การจัดตำแหน่ง target: หน้าต่าง [t+7L, t+7L+6] ต้องตรงเป๊ะ
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    flag = pd.Series(0.0, index=idx)
    flag.iloc[30] = 1.0  # เหตุการณ์วันที่ index 30
    t = weekly_event_targets(flag, leads=[2])["lead2"]
    # issue date t มี y=1 เมื่อ 30 อยู่ใน [t+14, t+20] -> t ใน [10, 16]
    hits = set(np.where(t == 1.0)[0].tolist())
    assert hits == set(range(10, 17)), f"ตำแหน่ง target ผิด: {sorted(hits)}"
    # ปลายอนุกรม: หน้าต่างไม่ครบ -> NaN
    assert t.iloc[-(14 + 6):].isna().all(), "หน้าต่างไม่ครบต้องเป็น NaN"
    print("[OK] target อยู่ตำแหน่ง [t+14, t+20] เป๊ะ และหน้าต่างไม่ครบ -> NaN")

    # 2) ช่องว่างข้อมูล (จำลองข้ามปี ส.ค.-ธ.ค. หาย) -> target ที่คร่อมช่องว่างเป็น NaN
    idx2 = idx.delete(slice(40, 60))  # เจาะรู 20 วัน
    flag2 = pd.Series(0.0, index=idx2)
    t2 = weekly_event_targets(flag2, leads=[2])["lead2"]
    # issue date ที่หน้าต่างคร่อมรู ต้องเป็น NaN
    bad = [i for i, ts in enumerate(idx2)
           if any((ts + pd.Timedelta(days=k)) not in set(idx2) for k in range(14, 21))]
    assert t2.iloc[bad].isna().all(), "หน้าต่างคร่อมช่องว่างต้องเป็น NaN"
    assert t2.notna().sum() > 0
    print("[OK] หน้าต่างที่คร่อมช่องว่างข้อมูล -> NaN (ถูกตัดตอนเทรน)")

    # 3) lookback features ไม่มองอนาคต: ขยับค่าวันสุดท้าย ไม่กระทบ feature วันก่อนหน้า
    daily = pd.DataFrame({
        "sm1": np.linspace(0.3, 0.4, 60), "sm3": np.linspace(0.35, 0.45, 60),
        "tmax_rm": np.linspace(30, 38, 60), "hot_rm": 0.0, "hw_rm": 0.0,
    }, index=pd.date_range("2020-01-01", periods=60, freq="D"))
    f_a = lookback_features(daily)
    daily2 = daily.copy()
    daily2.iloc[-1, daily2.columns.get_loc("sm1")] = 9.9
    f_b = lookback_features(daily2)
    same = f_a.iloc[:-1].drop(columns=["doy"]).fillna(-1) == \
        f_b.iloc[:-1].drop(columns=["doy"]).fillna(-1)
    assert same.all().all(), "feature ย้อนหลังห้ามเปลี่ยนเมื่อแก้ข้อมูลอนาคต (leakage!)"
    print("[OK] features มองย้อนหลังเท่านั้น (แก้ค่าอนาคตแล้ว feature อดีตไม่ขยับ)")

    # 4) regional mean ถ่วง cos(lat): แถบ lat สูงน้ำหนักต้องน้อยกว่า
    da = xr.DataArray(
        np.array([[[1.0, 1.0], [3.0, 3.0]]]),  # (time=1, latitude=2, longitude=2)
        dims=["time", "latitude", "longitude"],
        coords={"time": [pd.Timestamp("2020-01-01")], "latitude": [80.0, 0.0],
                "longitude": [100.0, 101.0]},
    )
    got = float(regional_mean(da).squeeze())
    w_hi, w_lo = np.cos(np.deg2rad(80.0)), 1.0
    expect = (1.0 * w_hi + 3.0 * w_lo) / (w_hi + w_lo)
    assert abs(got - expect) < 1e-9, (got, expect)
    print(f"[OK] regional mean ถ่วง cos(lat): {got:.4f} = ค่าคาด {expect:.4f}")

    # 5) Niño lag: ค่าของวันใดๆ ต้องเท่ากับ anomaly ของ 'เดือนก่อนหน้า'
    tmp = Path(__file__).resolve().parent / "_tmp_nino_test.csv"
    months = pd.date_range("2019-11-01", periods=6, freq="MS")
    pd.DataFrame({"date": months, "nino34_anom": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
                 ).to_csv(tmp, index=False)
    try:
        days = pd.DatetimeIndex(["2020-01-15", "2020-02-01", "2020-03-31"])
        got_n = nino_lagged_daily(tmp, days)
        # ม.ค.2020 -> ใช้ค่า ธ.ค.2019 (0.6), ก.พ. -> ม.ค. (0.7), มี.ค. -> ก.พ. (0.8)
        assert np.allclose(got_n.values, [0.6, 0.7, 0.8]), got_n.values
    finally:
        tmp.unlink(missing_ok=True)
    print("[OK] Niño3.4 lag 1 เดือนถูกต้อง")

    print("[OK] self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        _selftest()
        print()
        build()
