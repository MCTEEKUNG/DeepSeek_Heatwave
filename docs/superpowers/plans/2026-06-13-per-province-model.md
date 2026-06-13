# Per-Province Pooled Sub-Seasonal Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** เพิ่มความน่าจะเป็น heatwave 2–6 สัปดาห์ **รายจังหวัด (77)** ด้วยโมเดล pooled ตัวเดียว + province features — เสริม ไม่แทน pipeline regional — แล้วออก `docs/forecast_provinces.json` ให้ webapp (#2) ใช้

**Architecture:** ดึง Tmax/soil ราย "cell ใกล้ centroid จังหวัด" จาก `.nc` grid เดิม, reuse กลไก per-cell heatwave (`heatwave_target`) + `lookback_features`/`mjo`/`nino` (`build_dataset`) ต่อจังหวัด, stack เป็น pooled frame (จังหวัด×วัน), train โมเดลเดียว, validate ด้วย **date-blocked** rolling-origin CV (กัน temporal+spatial leakage). ตัวสร้าง feature ตัวเดียว (`build_provinces_features`) ใช้ทั้ง train และ serve (parity).

**Tech Stack:** Python 3.12, xarray, pandas, numpy, scikit-learn, lightgbm, **pyarrow (ใหม่)**; inline `_selftest()` รันผ่าน `python scripts/<m>.py [test]` (ธรรมเนียม repo — ไม่ใช่ pytest สำหรับ scripts/)

**สเปก:** `docs/superpowers/specs/2026-06-13-per-province-model-design.md`

---

## Prerequisites
- branch `feat/per-province-model` checkout แล้ว (spec commit อยู่บนนี้)
- prefix รันด้วย `PYTHONIOENCODING=utf-8`
- raw `.nc` (Tmax 60, soil 120) + indices มีครบ ; ไม่ดาวน์โหลดใหม่

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `data/provinces.csv` | 77 จังหวัด: id, code, name_th, name_en, region, lat, lon (พอร์ตจาก Heatwave_AI) |
| Modify | `.gitignore` | ยกเว้น `!data/provinces.csv` ให้ track ได้ |
| Create | `scripts/province_grid.py` | centroid→nearest cell + ดึงอนุกรมราย cell จังหวัด |
| Modify | `requirements.txt` | เพิ่ม `pyarrow` |
| Create | `scripts/build_provinces_dataset.py` | `build_provinces_features()` (reuse train+serve) + `build()` → parquet |
| Create | `scripts/train_provinces.py` | date-blocked CV, pooled train, baseline, pooled+per-province BSS, gate |
| Create | `scripts/analysis/leak_check_r1_provinces.py` | R1 gate บน pooled (lead 2) |
| Create | `scripts/train_final_provinces.py` | train pooled prod model ราย lead → .pkl |
| Create | `scripts/predict_provinces.py` | latest forecast รายจังหวัด → `docs/forecast_provinces.json` |

---

## Task 1: provinces.csv + province_grid.py

**Files:**
- Create: `data/provinces.csv`
- Modify: `.gitignore`
- Create: `scripts/province_grid.py`

- [ ] **Step 1: พอร์ต provinces.csv + ยกเว้น gitignore**

```bash
cp /c/Users/ASUS/Heatwave_AI/data/provinces.csv /c/Users/ASUS/DeepSeek_Heatwave/data/provinces.csv
head -1 data/provinces.csv   # ต้องเป็น: id,code,name_th,name_en,region,lat,lon
```
แก้ `.gitignore` — เพิ่มบรรทัดใต้ `data/`:
```
data/
!data/provinces.csv
```

- [ ] **Step 2: เขียน failing self-test ใน `scripts/province_grid.py`** (สร้างไฟล์ ใส่เฉพาะบล็อก `__main__` ก่อน)

```python
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import numpy as np
    import pandas as pd
    import xarray as xr

    # 1) โหลด provinces ครบ 77 + region อยู่ในชุดที่รู้จัก
    pv = load_provinces()
    assert len(pv) == 77, len(pv)
    assert set(pv["region"]) <= set(REGIONS), set(pv["region"]) - set(REGIONS)
    print(f"[OK] provinces.csv: {len(pv)} จังหวัด, regions={sorted(set(pv['region']))}")

    # 2) nearest_cell บนกริดสังเคราะห์ที่ตรงกริดจริง (lat 5..21, lon 97..106, 0.25°)
    lat = np.arange(5.0, 21.0001, 0.25)[::-1]
    lon = np.arange(97.0, 106.0001, 0.25)
    da = xr.DataArray(np.zeros((1, len(lat), len(lon))), dims=["time", "latitude", "longitude"],
                      coords={"time": [pd.Timestamp("2020-01-01")], "latitude": lat, "longitude": lon})
    cy, cx = nearest_cell(da, 13.7563, 100.5018)  # กรุงเทพฯ
    assert abs(cy - 13.75) < 1e-9 and abs(cx - 100.50) < 1e-9, (cy, cx)
    # 77 จังหวัด -> 76 cell unique (1 คู่ใช้ร่วม)
    cells = {nearest_cell(da, r.lat, r.lon) for r in pv.itertuples()}
    assert len(cells) == 76, len(cells)
    print(f"[OK] nearest_cell: BKK->({cy},{cx}); 77 จังหวัด -> {len(cells)} cell unique")

    # 3) province_series คืน Series รายวัน index=date ของ cell
    da2 = da.assign_coords(time=pd.date_range("2020-01-01", periods=1))
    s = province_series(da.isel(time=[0]).assign_coords(time=[pd.Timestamp("2020-01-01")]), 13.7563, 100.5018)
    assert isinstance(s, pd.Series) and len(s) == 1
    print("[OK] province_series คืน pandas Series")
    print("[OK] self-test ผ่าน")
```

Run: `PYTHONIOENCODING=utf-8 python scripts/province_grid.py`
Expected: FAIL — `NameError: name 'load_provinces' is not defined`

- [ ] **Step 3: เขียน implementation** — ใส่ส่วนบนของ `scripts/province_grid.py` (เหนือ `if __name__`)

```python
# scripts/province_grid.py
"""Map 77 Thai provinces -> nearest 0.25° ERA5 grid cell + extract per-province series.

centroid จาก data/provinces.csv (พอร์ตจาก Heatwave_AI) -> cell ใกล้สุดในกริด ERA5
(lat 5-21 / lon 97-106). 77 จังหวัด -> 76 unique cell (กทม./ปริมณฑล 1 คู่ใช้ cell ร่วม).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
PROVINCES_CSV = ROOT / "data" / "provinces.csv"
REGIONS = ["Central", "North", "Northeast", "East", "West", "South"]


def load_provinces(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or PROVINCES_CSV)
    need = {"id", "code", "name_th", "name_en", "region", "lat", "lon"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"provinces.csv ขาดคอลัมน์: {missing}")
    return df


def _lat_lon_names(da: xr.DataArray) -> tuple[str, str]:
    latn = "latitude" if "latitude" in da.coords else "lat"
    lonn = "longitude" if "longitude" in da.coords else "lon"
    return latn, lonn


def nearest_cell(da: xr.DataArray, lat: float, lon: float) -> tuple[float, float]:
    """พิกัด (lat, lon) ของ cell ที่ใกล้ centroid ที่สุดในกริดของ da."""
    latn, lonn = _lat_lon_names(da)
    la = np.asarray(da[latn].values, dtype=float)
    lo = np.asarray(da[lonn].values, dtype=float)
    return float(la[np.abs(la - lat).argmin()]), float(lo[np.abs(lo - lon).argmin()])


def province_series(da: xr.DataArray, lat: float, lon: float) -> pd.Series:
    """อนุกรมรายวันของ cell ใกล้ centroid -> pandas Series (index=date normalize)."""
    latn, lonn = _lat_lon_names(da)
    cell = da.sel({latn: lat, lonn: lon}, method="nearest")
    idx = pd.DatetimeIndex(cell["time"].values).normalize()
    return pd.Series(np.asarray(cell.values, dtype=float), index=idx)
```

- [ ] **Step 4: รัน self-test ให้ผ่าน**

Run: `PYTHONIOENCODING=utf-8 python scripts/province_grid.py`
Expected: PASS — เห็น `[OK] provinces.csv: 77 ...`, `[OK] nearest_cell: BKK->(13.75,100.5); 77 จังหวัด -> 76 cell unique`, `[OK] self-test ผ่าน`

- [ ] **Step 5: commit**

```bash
git add data/provinces.csv .gitignore scripts/province_grid.py
git commit -m "$(printf 'feat: province centroids + nearest-cell mapping (per-province #1)\n\nPort provinces.csv from Heatwave_AI; map each of 77 provinces to the\nnearest 0.25 deg ERA5 cell (76 unique; 1 BKK-area collision).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 2: build_provinces_dataset.py — pooled feature builder + dataset

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/build_provinces_dataset.py`

- [ ] **Step 1: เพิ่ม pyarrow + ติดตั้ง**

แก้ `requirements.txt` เพิ่มบรรทัด (ใต้ lightgbm):
```
pyarrow==22.0.0
```
Run: `python -m pip install "pyarrow==22.0.0" && python -c "import pyarrow; print('pyarrow', pyarrow.__version__)"`
Expected: พิมพ์เวอร์ชัน ไม่ error (ถ้าเวอร์ชัน pin ไม่มี ใช้ `pip install pyarrow` แล้วอัปเดต pin เป็นเวอร์ชันที่ลงได้)

- [ ] **Step 2: เขียน failing self-test** — สร้าง `scripts/build_provinces_dataset.py` ใส่เฉพาะบล็อก `__main__`:

```python
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
```
และฟังก์ชัน `_selftest()` (วางเหนือ `__main__`):
```python
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
    # province-static + region one-hot ครบ
    pv = load_provinces()
    cols = province_static_columns()
    assert cols == ["lat", "lon"] + [f"region_{r}" for r in REGIONS], cols
    print(f"[OK] leakage-safe local features + province_static {len(cols)} คอลัมน์")
    print("[OK] self-test ผ่าน")
```

Run: `PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test`
Expected: FAIL — `NameError`/`ImportError` (ฟังก์ชันยังไม่นิยาม)

- [ ] **Step 3: เขียน implementation** — ส่วนบนของ `scripts/build_provinces_dataset.py`:

```python
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
LOCAL_FEATURES = ["sm1", "sm1_mean7", "sm1_mean30", "sm1_trend",
                  "sm3", "sm3_mean7", "sm3_mean30", "sm3_trend",
                  "tmax_rm", "tmax_mean7", "in_hw_today", "hot_frac7"]
SHARED_FEATURES = ["mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_sin", "mjo_cos",
                   "nino34_lag1m", "doy_sin", "doy_cos"]


def province_static_columns() -> list[str]:
    return ["lat", "lon"] + [f"region_{r}" for r in REGIONS]


def _soil_grid(layer: int) -> xr.DataArray:
    files = sorted(SOIL_DIR.glob(f"era5_sm_l{layer}_thailand_*.nc"))
    if not files:
        raise FileNotFoundError(f"ไม่พบ soil moisture ชั้น {layer} ใน {SOIL_DIR}")
    ds = xr.open_mfdataset([str(p) for p in files], combine="by_coords")
    da = ds[f"swvl{layer}"].load()
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})
    if "number" in da.coords:
        da = da.drop_vars("number")
    return da.sortby("time")


def build_provinces_features(verbose: bool = True) -> tuple[pd.DataFrame, xr.DataArray]:
    """คืน (feat_all, hw_grid). feat_all = pooled features ทุกจังหวัด (ยังไม่มี target).

    reuse ตัวเดียวทั้ง train และ serve เพื่อ parity. hw_grid = heatwave per-cell (ไว้ทำ target ใน build()).
    """
    def log(m):
        if verbose:
            print(m, flush=True)
    log("[prov] โหลด Tmax grid + เกณฑ์ p90 ราย doy ราย cell ...")
    t_grid = load_tmax_celsius(sorted(TMAX_DIR.glob("era5_tmax_thailand_*.nc")))
    thr90 = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)
    hot_grid = hot_days(t_grid, thr90)
    hw_grid = flag_heatwaves(hot_grid, min_len=MIN_RUN)
    log("[prov] โหลด soil moisture grid ชั้น 1, 3 ...")
    sm1_grid, sm3_grid = _soil_grid(1), _soil_grid(3)
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
        feat = lookback_features(daily)                      # local (รวม in_hw_today trailing, hot_frac7, doy*)
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
    return feat_all, hw_grid


def build(verbose: bool = True) -> pd.DataFrame:
    feat_all, hw_grid = build_provinces_features(verbose=verbose)
    pv = load_provinces()
    # targets ราย lead จาก heatwave ของ cell แต่ละจังหวัด
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
```

- [ ] **Step 4: รัน self-test ให้ผ่าน**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test`
Expected: PASS — `[OK] leakage-safe local features + province_static 8 คอลัมน์`, `[OK] self-test ผ่าน`

- [ ] **Step 5: build dataset จริง (อ่าน grid ~นาที, ใช้ RAM พอควร)**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py`
Expected: `[OK] .../dataset_provinces.parquet rows=~828000` + base_rate ราย lead (~0.13 ใกล้ regional). บันทึกตัวเลขไว้

- [ ] **Step 6: commit** (parquet อยู่ใต้ data/ = gitignore ไม่ commit)

```bash
git add scripts/build_provinces_dataset.py requirements.txt
git commit -m "$(printf 'feat: pooled per-province dataset builder (reuses regional machinery)\n\nbuild_provinces_features() shared by train + serve (parity); per-cell\nheatwave label per province; province-static lat/lon/region one-hot.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 3: train_provinces.py — date-blocked CV + gate + per-province BSS

**Files:**
- Create: `scripts/train_provinces.py`

- [ ] **Step 1: เขียน failing self-test** — สร้าง `scripts/train_provinces.py` ใส่ `__main__` + `_selftest`:

```python
def _selftest() -> None:
    """date-blocked folds: train ทุกแถว 'ก่อน' test ตามเวลา, มี gap, ไม่มีวันทับกัน."""
    dates = pd.date_range("2000-01-01", periods=365 * 6, freq="D")
    df = pd.DataFrame({"date": np.repeat(dates, 3)})  # 3 จังหวัดจำลอง/วัน
    folds = list(date_blocked_folds(df["date"], n_splits=4, test_size=28, gap=42))
    assert len(folds) == 4, len(folds)
    for tr_dates, te_dates in folds:
        assert max(tr_dates) < min(te_dates), "train ต้องมาก่อน test (ตามวัน)"
        gap = (min(te_dates) - max(tr_dates)).days - 1
        assert gap >= 42, f"gap {gap} < 42"
        assert tr_dates.isdisjoint(te_dates), "วัน train/test ห้ามทับ"
    print("[OK] date-blocked folds: train ก่อน test, gap พอ, ไม่ทับ")
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
        main()
```

Run: `PYTHONIOENCODING=utf-8 python scripts/train_provinces.py test`
Expected: FAIL — `NameError: name 'date_blocked_folds' is not defined`

- [ ] **Step 2: เขียน implementation** — ส่วนบนของ `scripts/train_provinces.py`:

```python
"""เทรน + ประเมินผลโมเดล pooled per-province (sub-seasonal heatwave 2-6 สัปดาห์).

CV = date-blocked rolling-origin (ทุกจังหวัดของช่วงวันเดียวกันอยู่ fold เดียวกัน -> กัน
temporal+spatial leakage). baseline = seasonal climatology + persistence 'รายจังหวัด'.
รายงาน pooled BSS เทียบ baseline + per-province BSS (guard n<50). reuse evaluate + train.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import RollingOriginCV
from evaluate import (evaluate_probabilistic, predict_seasonal_climatology,
                      persistence_probs, brier_skill_score)
from train import FEATURES as REGIONAL_FEATURES, make_estimator, fit_predict_calibrated
from build_provinces_dataset import OUT_FILE, province_static_columns
from train_final import PROD_MODEL

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
ANALYSIS_DIR = OUT_DIR / "analysis"
LEADS = [2, 3, 4, 5, 6]
PRIMARY_TARGET = "y_rm"
GAP = 49
N_SPLITS = 5
TEST_SIZE = 300            # วัน (เหมือน regional)
PER_PROVINCE_MIN_N = 50    # guard: จังหวัด/ที่ n<50 = ไม่น่าเชื่อถือ
FEATURES_P = list(REGIONAL_FEATURES) + province_static_columns()


def date_blocked_folds(dates, n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP):
    """yield (train_dates:set, test_dates:set) แบ่งตาม 'วันที่ไม่ซ้ำ' ด้วย RollingOriginCV."""
    uniq = np.array(sorted(pd.unique(pd.DatetimeIndex(dates))))
    cv = RollingOriginCV(n_splits=n_splits, test_size=test_size, gap=gap, expanding=True)
    for tr_idx, te_idx in cv.split(len(uniq)):
        yield set(pd.DatetimeIndex(uniq[tr_idx])), set(pd.DatetimeIndex(uniq[te_idx]))


def _baseline_by_province(sub_tr, sub_te, col):
    """climatology + persistence รายจังหวัด: เรียนจาก train ของแต่ละจังหวัด -> map ให้ test."""
    p_clim = np.full(len(sub_te), np.nan)
    p_pers = np.full(len(sub_te), np.nan)
    te_reset = sub_te.reset_index(drop=True)
    for pid, g_te in te_reset.groupby("province_id"):
        g_tr = sub_tr[sub_tr["province_id"] == pid]
        if g_tr.empty:
            continue
        idx = g_te.index.to_numpy()
        p_clim[idx] = predict_seasonal_climatology(
            g_tr["doy"].to_numpy(int), g_tr[col].to_numpy(float), g_te["doy"].to_numpy(int))
        p_pers[idx] = persistence_probs(
            g_tr["in_hw_today"].to_numpy(int), g_tr[col].to_numpy(float),
            g_te["in_hw_today"].to_numpy(int))
    return p_clim, p_pers


def run_lead(df, lead, verbose=True):
    """CV pooled ของ lead เดียว -> (pooled_metrics_row, per_province_df, preds_df)."""
    col = f"{PRIMARY_TARGET}_l{lead}"
    sub = df[FEATURES_P + ["doy", "province_id", "date", col]].dropna().sort_values("date")
    preds = []
    for tr_dates, te_dates in date_blocked_folds(sub["date"]):
        tr = sub[sub["date"].isin(tr_dates)]
        te = sub[sub["date"].isin(te_dates)]
        if len(tr) == 0 or len(te) == 0 or tr[col].nunique() < 2:
            continue
        p = fit_predict_calibrated(PROD_MODEL, tr[FEATURES_P].to_numpy(float),
                                   tr[col].to_numpy(float), te[FEATURES_P].to_numpy(float))
        if p is None:
            continue
        p_clim, _p_pers = _baseline_by_province(tr, te, col)
        block = te[["province_id", "date", col]].copy()
        block["p"] = p
        block["p_clim"] = p_clim
        preds.append(block)
    pred = pd.concat(preds, ignore_index=True).dropna(subset=["p_clim"])
    pooled = evaluate_probabilistic(pred[col].to_numpy(), pred["p"].to_numpy(),
                                    baseline_prob=pred["p_clim"].to_numpy())
    pooled = {"lead": lead, **pooled}
    # per-province BSS (guard n<50)
    rows = []
    for pid, g in pred.groupby("province_id"):
        n = len(g)
        bss = brier_skill_score(g[col].to_numpy(), g["p"].to_numpy(),
                                baseline_prob=g["p_clim"].to_numpy()) if n >= 2 else float("nan")
        rows.append({"province_id": pid, "lead": lead, "n": n, "bss": bss,
                     "reliable": n >= PER_PROVINCE_MIN_N})
    per_prov = pd.DataFrame(rows)
    if verbose:
        print(f"  lead {lead}: n={len(pred)}, pooled BSS={pooled['bss']:+.3f}, AUC={pooled['auc']:.3f}", flush=True)
    return pooled, per_prov, pred


def main() -> int:
    df = pd.read_parquet(OUT_FILE)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== train_provinces: {len(df)} แถว | {PROD_MODEL}_cal | leads={LEADS} ===")
    pooled_rows, per_prov_all = [], []
    for lead in LEADS:
        pooled, per_prov, _pred = run_lead(df, lead)
        pooled_rows.append(pooled)
        per_prov_all.append(per_prov)
    pooled_df = pd.DataFrame(pooled_rows)
    per_prov_df = pd.concat(per_prov_all, ignore_index=True)
    pooled_df.to_csv(ANALYSIS_DIR / "provinces_pooled_bss.csv", index=False)
    per_prov_df.to_csv(ANALYSIS_DIR / "provinces_per_province_bss.csv", index=False)
    n_win = int((pooled_df["bss"] > 0).sum())
    print(f"\npooled BSS>0: {n_win}/{len(LEADS)} leads")
    print(pooled_df.round(3).to_string(index=False))
    print(f"[OK] ผลที่ {ANALYSIS_DIR}/provinces_*.csv")
    return 0
```

- [ ] **Step 3: รัน self-test ให้ผ่าน**

Run: `PYTHONIOENCODING=utf-8 python scripts/train_provinces.py test`
Expected: PASS — `[OK] date-blocked folds: train ก่อน test, gap พอ, ไม่ทับ`

- [ ] **Step 4: รันจริง (pooled CV — เป็นนาที)**

Run: `PYTHONIOENCODING=utf-8 python scripts/train_provinces.py`
Expected: ตาราง pooled BSS ราย lead + `pooled BSS>0: N/5` ; เขียน `provinces_pooled_bss.csv` + `provinces_per_province_bss.csv`. **Gate:** pooled BSS>0 ควรได้เกือบทุก lead (รายงานตรงๆ ตามจริง)

- [ ] **Step 5: commit**

```bash
git add scripts/train_provinces.py
git commit -m "$(printf 'feat: pooled per-province training + date-blocked CV + per-province BSS\n\nDate-blocked rolling-origin CV (guards temporal+spatial leakage);\nper-province seasonal-climatology and persistence baselines; pooled\nand per-province BSS with n<50 reliability guard.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 4: R1 leak gate on pooled (lead 2)

**Files:**
- Create: `scripts/analysis/leak_check_r1_provinces.py`

- [ ] **Step 1: เขียนสคริปต์ gate** — สร้าง `scripts/analysis/leak_check_r1_provinces.py`:

```python
"""R1 leak gate บน pooled per-province (lead 2) — วัด ΔBSS แบบ leave-block-out.

baked = label ที่เกณฑ์ p90 fit จากทุกปี (ติด leak) ; leakfree = fit threshold per-cell จาก
'วันใน train-fold เท่านั้น'. คาด ΔBSS เล็ก (หักล้างใน BSS ratio เหมือน regional). lead 2 (headline).
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from heatwave_target import load_tmax_celsius, doy_window_percentile, hot_days, flag_heatwaves
from build_dataset import weekly_event_targets, MIN_RUN, PCTL_WINDOW, TMAX_DIR
from province_grid import load_provinces, province_series
from build_provinces_dataset import OUT_FILE
from train_provinces import FEATURES_P, date_blocked_folds, _baseline_by_province
from train import fit_predict_calibrated
from train_final import PROD_MODEL
from evaluate import brier_skill_score

LEAD = 2


def pooled_bss(df, t_grid, pv, leakfree):
    col = f"y_rm_l{LEAD}"
    sub = df[FEATURES_P + ["doy", "province_id", "date", col]].dropna().sort_values("date")
    preds = []
    for tr_dates, te_dates in date_blocked_folds(sub["date"]):
        if leakfree:
            cutoff = np.datetime64(max(tr_dates))
            thr = doy_window_percentile(t_grid.sel(time=slice(None, cutoff)), q=90, window=PCTL_WINDOW)
            hw = flag_heatwaves(hot_days(t_grid, thr), min_len=MIN_RUN).astype(float)
            relabel = []
            for r in pv.itertuples():
                tg = weekly_event_targets(province_series(hw, r.lat, r.lon).sort_index(), [LEAD])
                relabel.append(pd.DataFrame({"province_id": int(r.id), "date": tg.index,
                                             "y_lf": tg[f"lead{LEAD}"].to_numpy()}))
            ymap = pd.concat(relabel, ignore_index=True)
            s = sub.merge(ymap, on=["province_id", "date"], how="left")
            s = s.dropna(subset=["y_lf"]); s[col] = s["y_lf"]
        else:
            s = sub
        tr = s[s["date"].isin(tr_dates)]; te = s[s["date"].isin(te_dates)]
        if len(tr) == 0 or len(te) == 0 or tr[col].nunique() < 2:
            continue
        p = fit_predict_calibrated(PROD_MODEL, tr[FEATURES_P].to_numpy(float),
                                   tr[col].to_numpy(float), te[FEATURES_P].to_numpy(float))
        if p is None:
            continue
        p_clim, _ = _baseline_by_province(tr, te, col)
        b = te[[col]].copy(); b["p"] = p; b["p_clim"] = p_clim
        preds.append(b)
    pred = pd.concat(preds, ignore_index=True).dropna(subset=["p_clim"])
    return brier_skill_score(pred[col].to_numpy(), pred["p"].to_numpy(), baseline_prob=pred["p_clim"].to_numpy())


def main() -> int:
    df = pd.read_parquet(OUT_FILE)
    t_grid = load_tmax_celsius(sorted(TMAX_DIR.glob("era5_tmax_thailand_*.nc")))
    pv = load_provinces()
    b = pooled_bss(df, t_grid, pv, leakfree=False)
    f = pooled_bss(df, t_grid, pv, leakfree=True)
    print(f"R1 gate (pooled, lead {LEAD}): BSS baked={b:+.4f} leakfree={f:+.4f} dBSS={f-b:+.4f}")
    print("เทียบ |dBSS| กับครึ่ง 95% CI ของ BSS (provinces) -> ใน CI = document, คง frozen-all-history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: รัน gate (เป็นนาที — train pooled 2 รอบ)**

Run: `PYTHONIOENCODING=utf-8 python scripts/analysis/leak_check_r1_provinces.py`
Expected: บรรทัด `R1 gate (pooled, lead 2): BSS baked=... leakfree=... dBSS=...`. บันทึก dBSS

- [ ] **Step 3: บันทึกผลต่อท้าย `docs/INTEGRITY.md`** — เพิ่มหัวข้อ:

```markdown

## R1 — per-province pooled (2026-06-13)
gate บน pooled per-province (lead 2): ΔBSS = <เติม> → <อยู่ใน/หลุด> 95% CI → <document/elimination>.
กลไกเดียวกับ regional (per-cell threshold) จึงคาดว่าหักล้างใน BSS ratio เช่นกัน.
```
เติมเลขจริงจาก Step 2 แทน `<...>`

- [ ] **Step 4: commit**

```bash
git add scripts/analysis/leak_check_r1_provinces.py docs/INTEGRITY.md
git commit -m "$(printf 'feat: R1 leak gate on pooled per-province (lead 2)\n\nLeave-block-out per-cell relabel vs baked; verdict in INTEGRITY.md.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 5: serving — train_final_provinces.py + predict_provinces.py → forecast_provinces.json

**Files:**
- Create: `scripts/train_final_provinces.py`
- Create: `scripts/predict_provinces.py`

- [ ] **Step 1: train_final_provinces.py** — สร้างไฟล์:

```python
"""เทรนโมเดล pooled per-province 'ตัวจบ' ราย lead -> models/heatwave_prov_lead{L}.pkl.

ใช้ขั้นตอนเดียวกับ train_provinces วัดผล (fit_calibrated_model) -> deploy ไม่ drift.
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train import fit_calibrated_model
from train_final import PROD_MODEL
from train_provinces import FEATURES_P, LEADS, PRIMARY_TARGET
from build_provinces_dataset import OUT_FILE

MODEL_DIR = ROOT / "models"


def artifact_path(lead: int) -> Path:
    return MODEL_DIR / f"heatwave_prov_lead{lead}.pkl"


def train_final(verbose: bool = True) -> list[Path]:
    df = pd.read_parquet(OUT_FILE)
    MODEL_DIR.mkdir(exist_ok=True)
    written = []
    for lead in LEADS:
        col = f"{PRIMARY_TARGET}_l{lead}"
        sub = df[FEATURES_P + [col]].dropna()
        X = sub[FEATURES_P].to_numpy(float)
        y = sub[col].to_numpy(float)
        fitted = fit_calibrated_model(PROD_MODEL, X, y)
        if fitted is None:
            raise RuntimeError(f"lead {lead}: เทรนไม่ได้ (ข้อมูลสั้น/คลาสเดียว)")
        est, cal = fitted
        art = {"estimator": est, "calibrator": cal, "features": list(FEATURES_P),
               "model_name": f"{PROD_MODEL}_cal", "lead_weeks": lead,
               "base_rate": float(y.mean()), "n_train": int(len(sub)),
               "trained_at": datetime.now(timezone.utc).isoformat()}
        with open(artifact_path(lead), "wb") as fh:
            pickle.dump(art, fh)
        written.append(artifact_path(lead))
        if verbose:
            print(f"  lead {lead}: n={len(sub)}, base_rate={art['base_rate']:.3f} -> {artifact_path(lead).name}")
    return written


if __name__ == "__main__":
    train_final()
```

- [ ] **Step 2: predict_provinces.py** — สร้างไฟล์ (reuse `build_provinces_features` -> parity ; reuse `predict.risk_level`):

```python
"""ออก forecast รายจังหวัด (lead 2-6) ของวันล่าสุดที่ feature ครบ -> docs/forecast_provinces.json."""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_provinces_dataset import build_provinces_features, FEATURES_P
from train_provinces import LEADS
from train_final_provinces import artifact_path
from predict import risk_level
from province_grid import load_provinces

OUT_FILE = ROOT / "docs" / "forecast_provinces.json"
SKILL_CSV = ROOT / "outputs" / "analysis" / "provinces_per_province_bss.csv"


def _load_arts():
    arts = {}
    for L in LEADS:
        with open(artifact_path(L), "rb") as fh:
            arts[L] = pickle.load(fh)
    return arts


def build_forecast() -> dict:
    feat_all, _hw = build_provinces_features(verbose=False)
    arts = _load_arts()
    pv = load_provinces().set_index("id")
    skill = pd.read_csv(SKILL_CSV) if SKILL_CSV.exists() else pd.DataFrame()
    provinces_out = []
    for pid, g in feat_all.groupby("province_id"):
        valid = g.dropna(subset=FEATURES_P)
        if valid.empty:
            continue
        row = valid.sort_values("date").iloc[-1]
        X = row[FEATURES_P].to_numpy(float).reshape(1, -1)
        info = pv.loc[pid]
        fcs = []
        for L in LEADS:
            a = arts[L]
            p = float(a["calibrator"].transform(a["estimator"].predict_proba(X)[:, 1])[0])
            th, en, ratio = risk_level(p, a["base_rate"])
            fcs.append({"lead_weeks": L, "probability": round(p, 4),
                        "climatology_base_rate": round(a["base_rate"], 4),
                        "ratio_vs_normal": round(ratio, 2),
                        "risk_level_th": th, "risk_level_en": en})
        provinces_out.append({"id": int(pid), "code": info["code"],
                              "name_th": info["name_th"], "name_en": info["name_en"],
                              "region": info["region"], "lat": float(info["lat"]),
                              "lon": float(info["lon"]),
                              "issue_date": str(pd.Timestamp(row["date"]).date()),
                              "forecasts": fcs})
    out = {"schema_version": 1, "model": arts[LEADS[0]]["model_name"],
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "n_provinces": len(provinces_out), "provinces": provinces_out}
    if not skill.empty:
        out["skill"] = skill.to_dict(orient="records")
    return out


def _selftest() -> None:
    """parity: feature รายจังหวัดที่ build_provinces_features สร้าง = ที่อยู่ใน dataset_provinces.parquet."""
    from build_provinces_dataset import OUT_FILE as DS
    if not DS.exists():
        print("[ข้าม] ไม่มี dataset_provinces.parquet — รัน build ก่อน"); return
    feat_all, _ = build_provinces_features(verbose=False)
    ds = pd.read_parquet(DS)
    key = ["province_id", "date"]
    m = feat_all.merge(ds, on=key, suffixes=("_a", "_b"))
    import numpy as np
    bad = 0
    for c in FEATURES_P:
        a = m[f"{c}_a"].to_numpy(float); b = m[f"{c}_b"].to_numpy(float)
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-8) | (np.isnan(a) & np.isnan(b))
        bad += int((~ok).sum())
    assert bad == 0, f"parity ไม่ตรง {bad} จุด"
    print(f"[OK] train/serve parity: {len(FEATURES_P)} feature x {len(m)} แถว ตรง dataset")
    print("[OK] self-test ผ่าน")


def predict(verbose: bool = True) -> dict:
    fc = build_forecast()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"[OK] {OUT_FILE} | {fc['n_provinces']} จังหวัด | model {fc['model']}")
    return fc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        predict()
```

- [ ] **Step 3: train final + parity test + ออก forecast**

```
PYTHONIOENCODING=utf-8 python scripts/train_final_provinces.py
PYTHONIOENCODING=utf-8 python scripts/predict_provinces.py test
PYTHONIOENCODING=utf-8 python scripts/predict_provinces.py
```
Expected: train เซฟ 5 .pkl ; parity self-test `[OK] train/serve parity: ... ตรง dataset` ; predict เขียน `docs/forecast_provinces.json` ครบ 77 จังหวัด

- [ ] **Step 4: ตรวจ forecast.json**

Run:
```
PYTHONIOENCODING=utf-8 python -c "import json;d=json.load(open('docs/forecast_provinces.json',encoding='utf-8'));print('provinces',d['n_provinces']);print('sample',d['provinces'][0]['name_en'],[f['probability'] for f in d['provinces'][0]['forecasts']])"
```
Expected: `provinces 77` + ความน่าจะเป็น 5 ค่า ของจังหวัดแรก

- [ ] **Step 5: commit** (models/*.pkl + forecast_provinces.json tracked)

```bash
git add scripts/train_final_provinces.py scripts/predict_provinces.py models/heatwave_prov_lead2.pkl models/heatwave_prov_lead3.pkl models/heatwave_prov_lead4.pkl models/heatwave_prov_lead5.pkl models/heatwave_prov_lead6.pkl docs/forecast_provinces.json
git commit -m "$(printf 'feat: per-province serving -> forecast_provinces.json (#1 done)\n\nPooled final models per lead; predict reuses build_provinces_features\n(train/serve parity self-test). Output feeds the webapp map (#2).\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 6: regression guard — regional pipeline ยังครบ

**Files:** (ไม่แก้โค้ด — verify เท่านั้น)

- [ ] **Step 1: รัน self-test ของ pipeline regional เดิม + ของใหม่**

Run:
```
PYTHONIOENCODING=utf-8 python scripts/build_dataset.py test
PYTHONIOENCODING=utf-8 python scripts/predict.py test
PYTHONIOENCODING=utf-8 python scripts/province_grid.py
PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test
PYTHONIOENCODING=utf-8 python scripts/train_provinces.py test
```
Expected: ทั้งหมด PASS — ยืนยันว่างาน per-province (additive) ไม่ทำ regional พัง

- [ ] **Step 2: (ถ้ามีแก้) commit** — ปกติไม่มีไฟล์เปลี่ยนใน task นี้ ; ถ้าเจอ regression ให้แก้แล้ว commit

---

## Acceptance criteria
1. `data/provinces.csv` tracked ; `province_grid.py` self-test: 77→76 cell
2. `dataset_provinces.parquet` ~828k แถว ; builder leakage self-test ผ่าน
3. `train_provinces.py`: pooled BSS เทียบ clim+persist + per-province BSS ; **gate** (pooled BSS>0 ตามที่รายงานจริง)
4. R1 gate (pooled lead 2): ΔBSS + verdict ใน INTEGRITY.md
5. `predict_provinces.py`: parity self-test ผ่าน ; `forecast_provinces.json` 77 จังหวัด × 5 lead
6. regional pipeline เดิม self-test ผ่านครบ (additive ไม่ทำพัง)

---

## Self-Review (ผู้เขียนแผนตรวจแล้ว)
- **Spec coverage:** §3→T1 ; §4-6→T2 (label per-cell, features, parquet, build_provinces_features) ; §7-8→T3 (pooled model, date-blocked CV, per-province BSS) ; R1 §4→T4 ; §9 output→T5 ; non-goal "ไม่แทน regional" →T6 guard
- **Placeholder scan:** โค้ดเต็มทุก step ; `<...>` ใน INTEGRITY (T4 S3) เป็นช่องผลลัพธ์ที่ T4 เติม
- **Type/Name consistency:** `FEATURES_P` (= regional FEATURES + `province_static_columns()`) นิยามใน train_provinces, reuse ใน leak_check/train_final/predict ; `build_provinces_features()` reuse train(T2)+serve(T5) parity ; `date_blocked_folds`, `_baseline_by_province`, `PROD_MODEL`, `fit_predict_calibrated`, `fit_calibrated_model`, `risk_level`, `weekly_event_targets`, `province_series` — ทั้งหมดมีจริง/นิยามใน task ก่อนหน้า
- **ความเสี่ยงที่รู้:** ตอน execute ให้รัน task ตามลำดับ (T2 ต้อง build dataset ก่อน T3/T4/T5) ; pooled CV/relabel เป็นนาที — ใช้ background/timeout ยาวได้ ; ตรวจ RAM ตอนโหลด grid (T2/T4 โหลด t_grid เต็ม)
