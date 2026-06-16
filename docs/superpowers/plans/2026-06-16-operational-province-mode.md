# Operational Province Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ให้ forecast รายจังหวัดออกพยากรณ์จากข้อมูล ERA5 ล่าสุด (issue_date ขยับจาก 2023-12-31 → ~2026-06-10) โดยคงมาตรฐาน train/serve parity เดิม.

**Architecture:** เพิ่ม `operational=` flag ใน `build_provinces_features` (อ่าน `raw_recent/` + เกณฑ์ p90 แช่แข็ง + MJO impute) — code path เดียวกับ train เพื่อกัน skew. Freeze climatology รายจังหวัด (thr90 ราย cell + per-province base_rate) ใน artifact ใหม่. แก้ banner ฝั่ง frontend ให้โชว์ตามอายุข้อมูล.

**Tech Stack:** Python 3.12 (numpy/pandas/xarray, pickle), pytest (`python -m pytest`, pattern `sys.path.insert`+bare import); TypeScript/vitest/bun (frontend).

**Spec:** `docs/superpowers/specs/2026-06-16-operational-province-mode-design.md`

---

## File Structure

- **Create** `scripts/freeze_provinces_climatology.py` — สร้าง `models/climatology_provinces.pkl` (thr90 grid + per-province base_rate)
- **Create** `scripts/test_operational_provinces.py` — pytest: freeze structure, base_rate per-province, operational parity
- **Modify** `scripts/build_provinces_dataset.py` — recent dirs + `_soil_grid(layer, soil_dir=)` + `build_provinces_features(verbose, operational=False)`
- **Modify** `scripts/predict_provinces.py` — arg `operational`, per-province base_rate, MJO warning, issue_date
- **Modify** (frontend) `services/forecastService.ts` — เพิ่ม `isHistoricalRun()` pure helper
- **Modify** (frontend) `components/forecast/HistoricalRunBanner.tsx` — รับ `generatedAt`, ใช้ `isHistoricalRun`
- **Modify** (frontend) `app/(tabs)/alerts.tsx` — ส่ง `generatedAt`
- **Create** (frontend) `services/isHistoricalRun.test.ts` — vitest
- **Modify** `docs/RUNBOOK.md` — เพิ่มขั้น operational province

---

## Task 1: Freeze per-province climatology

**Files:**
- Create: `scripts/freeze_provinces_climatology.py`
- Create: `scripts/test_operational_provinces.py`
- Artifact (generated): `models/climatology_provinces.pkl`

- [ ] **Step 1: Write `freeze_provinces_climatology.py`**

```python
"""แช่แข็ง climatology รายจังหวัด -> models/climatology_provinces.pkl.

thr90_grid: เกณฑ์ p90 ราย doy ราย cell (จาก grid 30 ปี) — ใช้คำนวณ hot-day
            ในโหมด operational (recompute จากหน้าต่าง ~70 วันไม่ได้).
base_rate : {province_id: {lead: rate}} per-province per-lead จาก dataset parquet.
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
from heatwave_target import load_tmax_celsius, doy_window_percentile
from build_dataset import TMAX_DIR, PCTL_WINDOW, MIN_RUN, LEADS
from build_provinces_dataset import OUT_FILE
from train_provinces import PRIMARY_TARGET
from province_grid import load_provinces

CLIM_FILE = ROOT / "models" / "climatology_provinces.pkl"


def freeze(verbose: bool = True) -> dict:
    if verbose:
        print("[freeze] โหลด Tmax grid + คำนวณ thr90 ราย doy ราย cell ...", flush=True)
    t_grid = load_tmax_celsius(sorted(TMAX_DIR.glob("era5_tmax_thailand_*.nc")))
    thr90 = doy_window_percentile(t_grid, q=90, window=PCTL_WINDOW)

    if verbose:
        print("[freeze] คำนวณ base_rate per-province per-lead จาก parquet ...", flush=True)
    df = pd.read_parquet(OUT_FILE)
    pv = load_provinces()
    base: dict[int, dict[int, float]] = {}
    for pid, g in df.groupby("province_id"):
        base[int(pid)] = {}
        for L in LEADS:
            col = f"{PRIMARY_TARGET}_l{L}"
            m = g[col].notna()
            base[int(pid)][int(L)] = float(g.loc[m, col].mean()) if m.any() else float("nan")

    art = {
        "thr90_grid": thr90,
        "base_rate": base,
        "n_provinces": int(len(pv)),
        "leads": list(LEADS),
        "window": PCTL_WINDOW,
        "min_run": MIN_RUN,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": str(TMAX_DIR),
    }
    CLIM_FILE.parent.mkdir(exist_ok=True)
    with open(CLIM_FILE, "wb") as fh:
        pickle.dump(art, fh)
    if verbose:
        print(f"[OK] {CLIM_FILE} | thr90 {dict(thr90.sizes)} | base_rate {len(base)} จังหวัด x {len(LEADS)} lead")
    return art


if __name__ == "__main__":
    freeze()
```

- [ ] **Step 2: Run the freeze (generates artifact)**

Run: `cd C:/Users/ASUS/DeepSeek_Heatwave && PYTHONIOENCODING=utf-8 python scripts/freeze_provinces_climatology.py`
Expected: `[OK] models/climatology_provinces.pkl | thr90 {'dayofyear': 366, 'latitude': 65, 'longitude': 37} | base_rate 77 จังหวัด x 5 lead`

- [ ] **Step 3: Write the failing test for artifact structure**

```python
# scripts/test_operational_provinces.py
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CLIM = ROOT / "models" / "climatology_provinces.pkl"


def _load_clim():
    if not CLIM.exists():
        pytest.skip("ยังไม่มี climatology_provinces.pkl — รัน freeze ก่อน")
    with open(CLIM, "rb") as fh:
        return pickle.load(fh)


def test_frozen_climatology_structure():
    art = _load_clim()
    assert set(art["thr90_grid"].dims) == {"dayofyear", "latitude", "longitude"}
    assert art["thr90_grid"].sizes["dayofyear"] == 366
    assert art["n_provinces"] == 77
    assert art["leads"] == [2, 3, 4, 5, 6]


def test_base_rate_per_province_in_range_and_varies():
    art = _load_clim()
    base = art["base_rate"]
    assert len(base) == 77
    vals_lead2 = [base[p][2] for p in base]
    for v in vals_lead2:
        assert 0.0 <= v <= 1.0
    # per-province ต้อง "ต่างกันจริง" (แก้ quirk pooled ที่ทุกจังหวัดเท่ากัน)
    assert len(set(round(v, 4) for v in vals_lead2)) > 1
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONIOENCODING=utf-8 python -m pytest scripts/test_operational_provinces.py -v`
Expected: 2 passed (`test_frozen_climatology_structure`, `test_base_rate_per_province_in_range_and_varies`)

- [ ] **Step 5: Commit**

```bash
git add scripts/freeze_provinces_climatology.py scripts/test_operational_provinces.py
git commit -m "feat: freeze per-province climatology (thr90 grid + per-province base_rate)"
```

---

## Task 2: Operational data path in `build_provinces_features`

**Files:**
- Modify: `scripts/build_provinces_dataset.py`

- [ ] **Step 1: Add recent dirs + frozen-climatology loader + parametrize `_soil_grid`**

แก้หัวไฟล์ — เพิ่มหลังบรรทัด `OUT_FILE = ...`:

```python
RECENT_TMAX_DIR = TMAX_DIR.parent.parent / "raw_recent" / "tmax_thailand"
RECENT_SOIL_DIR = SOIL_DIR.parent.parent / "raw_recent" / "soil_moisture_thailand"
CLIM_PROV_FILE = ROOT / "models" / "climatology_provinces.pkl"


def _load_frozen_climatology() -> dict:
    import pickle
    if not CLIM_PROV_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบ {CLIM_PROV_FILE.name} — รัน `python scripts/freeze_provinces_climatology.py` ก่อน"
        )
    with open(CLIM_PROV_FILE, "rb") as fh:
        return pickle.load(fh)
```

แก้ `_soil_grid` ให้รับ `soil_dir`:

```python
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
```

- [ ] **Step 2: Add `operational=` to `build_provinces_features`**

แทนที่หัวฟังก์ชัน `build_provinces_features` (ส่วนโหลด grid + thr90 + soil) ด้วย:

```python
def build_provinces_features(verbose: bool = True, operational: bool = False) -> tuple[pd.DataFrame, xr.DataArray | None]:
    """คืน (feat_all, hw_grid). operational=True -> อ่าน raw_recent/ + thr90 แช่แข็ง
    (hw_grid=None เพราะ serve ไม่ทำ target). reuse builder เดียวกับ train เพื่อ parity."""
    def log(m):
        if verbose:
            print(m, flush=True)
    tmax_dir = RECENT_TMAX_DIR if operational else TMAX_DIR
    soil_dir = RECENT_SOIL_DIR if operational else SOIL_DIR
    log(f"[prov] โหลด Tmax grid ({'recent' if operational else 'full'}) ...")
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
```

(ส่วน `frames = []` … loop … `feat_all = pd.concat(...)` คงเดิมทั้งหมด ไม่แก้.)

ก่อน `return feat_all, hw_grid` (ปลายฟังก์ชัน) แทรก MJO impute สำหรับ operational:

```python
    if operational:
        from predict import impute_neutral_mjo, load_climatology
        means = (load_climatology() or {}).get("mjo_means")
        tmp = feat_all.set_index("date")
        tmp, imputed = impute_neutral_mjo(tmp, means)
        feat_all = tmp.reset_index()
        feat_all.attrs["mjo_imputed_dates"] = imputed
    log(f"[prov] features: {len(feat_all)} แถว x {len(pv)} จังหวัด")
    return feat_all, hw_grid
```

- [ ] **Step 3: Verify non-operational path unchanged (regression self-test)**

Run: `PYTHONIOENCODING=utf-8 python scripts/build_provinces_dataset.py test`
Expected: `[OK] leakage-safe local features ...` + `[OK] self-test ผ่าน` (ไม่พัง)

- [ ] **Step 4: Verify operational features build (smoke)**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "import sys; sys.path.insert(0,'scripts'); from build_provinces_dataset import build_provinces_features as b; f,h=b(verbose=True, operational=True); print('rows', len(f), '| latest date', str(f['date'].max())[:10], '| hw_grid', h)"
```
Expected: พิมพ์ `rows <N> | latest date 2026-06-10 | hw_grid None` (วันที่อยู่ปี 2026)

- [ ] **Step 5: Commit**

```bash
git add scripts/build_provinces_dataset.py
git commit -m "feat: operational path in build_provinces_features (raw_recent + frozen thr90 + MJO impute)"
```

---

## Task 3: Operational parity self-test

**Files:**
- Modify: `scripts/test_operational_provinces.py`

- [ ] **Step 1: Write failing parity test**

เพิ่มท้าย `scripts/test_operational_provinces.py`:

```python
import numpy as np


def test_operational_parity_with_historical():
    """feature ที่ operational path สร้าง = ที่ historical path สร้าง สำหรับวันที่ทับกัน
    (พิสูจน์ว่า operational mode ไม่ทำ feature เพี้ยน). ต้องมี raw_recent/ + frozen clim."""
    if not CLIM.exists():
        pytest.skip("ต้องมี climatology_provinces.pkl")
    from build_provinces_dataset import build_provinces_features, RECENT_TMAX_DIR, FEATURES_NONE  # noqa
    if not list(RECENT_TMAX_DIR.glob("*.nc")):
        pytest.skip("ยังไม่ได้ดึง raw_recent/ — รัน download ... recent ก่อน")
    from train_provinces import FEATURES_P
    feat_op, _ = build_provinces_features(verbose=False, operational=True)
    feat_hi, _ = build_provinces_features(verbose=False, operational=False)
    key = ["province_id", "date"]
    m = feat_op.merge(feat_hi, on=key, suffixes=("_op", "_hi"))
    assert len(m) > 0, "ไม่มีวันทับกันระหว่าง recent กับ historical grid"
    bad = 0
    for c in FEATURES_P:
        a = m[f"{c}_op"].to_numpy(float); b = m[f"{c}_hi"].to_numpy(float)
        ok = np.isclose(a, b, rtol=1e-6, atol=1e-8) | (np.isnan(a) & np.isnan(b))
        bad += int((~ok).sum())
    assert bad == 0, f"operational parity ไม่ตรง {bad} จุด"
```

(ลบ import `FEATURES_NONE` ออก — ใส่มาเพื่อให้ test fail ก่อนในขั้นนี้.)

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONIOENCODING=utf-8 python -m pytest scripts/test_operational_provinces.py::test_operational_parity_with_historical -v`
Expected: FAIL (`ImportError: cannot import name 'FEATURES_NONE'`)

- [ ] **Step 3: Fix the import line**

แก้บรรทัด import เป็น:
```python
    from build_provinces_dataset import build_provinces_features, RECENT_TMAX_DIR
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONIOENCODING=utf-8 python -m pytest scripts/test_operational_provinces.py -v`
Expected: 3 passed
หมายเหตุ: recent (2026-04-01..06-10) ทับกับ historical (..2023-12-31) เฉพาะถ้า window ซ้อน — ถ้าไม่ทับ test จะ skip ที่ `len(m) > 0`. กรณีนี้ดึง recent ของช่วงที่อยู่ในปี grid เต็มเพื่อทดสอบ parity จริง: ตั้ง env หรือใช้ปีที่ทับ. ถ้า recent ปัจจุบันไม่ทับ ให้ยอมรับ skip + พึ่ง parity เดิมของ historical (Task 5 regression) — operational ใช้ builder/`lookback_features` ตัวเดียวกัน ต่างแค่ source dir + thr90 แช่แข็ง.

- [ ] **Step 5: Commit**

```bash
git add scripts/test_operational_provinces.py
git commit -m "test: operational/historical feature parity"
```

---

## Task 4: `predict_provinces.py operational` + per-province base_rate

**Files:**
- Modify: `scripts/predict_provinces.py`

- [ ] **Step 1: Load frozen base_rate + add operational to `build_forecast`**

แก้ `predict_provinces.py`. เพิ่ม import + loader ใต้ import เดิม:

```python
from build_provinces_dataset import _load_frozen_climatology
```

แก้ `build_forecast` ให้รับ `operational` และใช้ per-province base_rate:

```python
def build_forecast(operational: bool = False) -> dict:
    feat_all, _hw = build_provinces_features(verbose=False, operational=operational)
    arts = _load_arts()
    clim = _load_frozen_climatology()
    prov_base = clim["base_rate"]                       # {pid: {lead: rate}}
    imputed = feat_all.attrs.get("mjo_imputed_dates", set())
    pv = load_provinces().set_index("id")
    skill = pd.read_csv(SKILL_CSV) if SKILL_CSV.exists() else pd.DataFrame()
    provinces_out = []
    warned = False
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
            br = float(prov_base[int(pid)][int(L)])      # per-province base_rate
            th, en, ratio = risk_level(p, br)
            fcs.append({"lead_weeks": L, "probability": round(p, 4),
                        "climatology_base_rate": round(br, 4),
                        "ratio_vs_normal": round(ratio, 2),
                        "risk_level_th": th, "risk_level_en": en})
        if pd.Timestamp(row["date"]) in imputed:
            warned = True
        provinces_out.append({"id": int(pid), "code": info["code"],
                              "name_th": info["name_th"], "name_en": info["name_en"],
                              "region": info["region"], "lat": float(info["lat"]),
                              "lon": float(info["lon"]),
                              "issue_date": str(pd.Timestamp(row["date"]).date()),
                              "forecasts": fcs})
    out = {"schema_version": 1, "model": arts[LEADS[0]]["model_name"],
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "n_provinces": len(provinces_out), "provinces": provinces_out}
    if warned:
        out["warnings"] = ["ข้อมูล MJO ไม่อัปเดตถึงวันออกพยากรณ์ — ใช้ค่า MJO กลางแทน (ผลอาจคลาดเคลื่อนเล็กน้อย)"]
    if not skill.empty:
        out["skill"] = skill.to_dict(orient="records")
    return out
```

- [ ] **Step 2: Wire `operational` through `predict()` + CLI**

แก้ `predict` และ `__main__`:

```python
def predict(verbose: bool = True, operational: bool = False) -> dict:
    fc = build_forecast(operational=operational)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        iss = fc["provinces"][0]["issue_date"] if fc["provinces"] else "?"
        print(f"[OK] {OUT_FILE} | {fc['n_provinces']} จังหวัด | model {fc['model']} | issue {iss}")
    return fc
```

แก้ `__main__`:
```python
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "operational":
        predict(operational=True)
    else:
        predict()
```

- [ ] **Step 3: Run operational predict end-to-end**

Run: `PYTHONIOENCODING=utf-8 python scripts/predict_provinces.py operational`
Expected: `[OK] ...forecast_provinces.json | 77 จังหวัด | model logistic_balanced_cal | issue 2026-06-10`

- [ ] **Step 4: Assert issue_date is current-year + per-province base_rate varies (e2e test)**

เพิ่มใน `scripts/test_operational_provinces.py`:

```python
import json


def test_operational_forecast_is_current_and_per_province():
    out = ROOT / "docs" / "forecast_provinces.json"
    if not out.exists():
        pytest.skip("ยังไม่ได้รัน predict_provinces.py operational")
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["n_provinces"] == 77
    issue_years = {p["issue_date"][:4] for p in d["provinces"]}
    assert issue_years == {"2026"}, f"issue_date ไม่ใช่ปี 2026: {issue_years}"
    # per-province base_rate ต้องต่างกันจริง (lead 2)
    br = {round([f for f in p["forecasts"] if f["lead_weeks"] == 2][0]["climatology_base_rate"], 4)
          for p in d["provinces"]}
    assert len(br) > 1, "base_rate ยังเป็น pooled (ค่าเดียวทุกจังหวัด)"
    # probability ทุกจังหวัด/lead ∈ (0,1) — regime sanity
    for p in d["provinces"]:
        for f in p["forecasts"]:
            assert 0.0 < f["probability"] < 1.0
```

Run: `PYTHONIOENCODING=utf-8 python -m pytest scripts/test_operational_provinces.py -v`
Expected: 4 passed

หมายเหตุ: ถ้า test ล้ม เพราะ issue_date ไม่ใช่ 2026 → `nino34.csv`/`mjo_rmm.csv` ไม่อัปเดตถึง 2026 ทำให้แถวล่าสุดถูก dropna. แก้โดยรัน `python scripts/download_indices.py` ก่อน (refresh MJO/Niño34) แล้วรัน predict operational ใหม่.

- [ ] **Step 5: Commit**

```bash
git add scripts/predict_provinces.py scripts/test_operational_provinces.py
git commit -m "feat: predict_provinces operational mode + per-province base_rate risk"
```

---

## Task 5: Regression — pipeline เดิมต้องไม่พัง

**Files:** (รันอย่างเดียว ไม่แก้โค้ด)

- [ ] **Step 1: รัน self-test เดิมทุกตัวที่เกี่ยวข้อง**

Run:
```bash
cd C:/Users/ASUS/DeepSeek_Heatwave && PYTHONIOENCODING=utf-8 bash -c '
python scripts/heatwave_target.py >/dev/null && echo OK heatwave_target &&
python scripts/province_grid.py >/dev/null && echo OK province_grid &&
python scripts/build_provinces_dataset.py test >/dev/null && echo OK build_provinces &&
python scripts/predict_provinces.py test 2>&1 | tail -2'
```
Expected: `OK heatwave_target` / `OK province_grid` / `OK build_provinces` / parity `[OK] self-test ผ่าน`

- [ ] **Step 2: รัน pytest ทั้ง suite**

Run: `PYTHONIOENCODING=utf-8 python -m pytest scripts/ -q`
Expected: ทุก test ผ่าน (validate_contract, publish_bridge, operational_provinces)

- [ ] **Step 3: รัน validate gate บนผล operational**

Run: `PYTHONIOENCODING=utf-8 python scripts/validate_contract.py`
Expected: `[OK] contract ผ่าน: 77 จังหวัด, schema v1` (exit 0)

- [ ] **Step 4: Commit (ถ้ามีไฟล์ artifact/ผลที่ตั้งใจ track)**

```bash
git add models/climatology_provinces.pkl
git commit -m "chore: add frozen per-province climatology artifact"
```
(หมายเหตุ: `docs/forecast_provinces.json` เป็น artifact ที่ publish_bridge จัดการ — commit เฉพาะเมื่อตั้งใจ publish.)

---

## Task 6: Frontend — banner conditional ตามอายุข้อมูล

**Files:**
- Modify: `services/forecastService.ts`
- Create: `services/isHistoricalRun.test.ts`
- Modify: `components/forecast/HistoricalRunBanner.tsx`
- Modify: `app/(tabs)/alerts.tsx`

(รันใน `C:/Users/ASUS/HeatMAP_Frontend`)

- [ ] **Step 1: Write failing vitest for pure helper**

```typescript
// services/isHistoricalRun.test.ts
import { describe, it, expect } from 'vitest';
import { isHistoricalRun } from './forecastService';

describe('isHistoricalRun', () => {
  it('true when issue_date is much older than generated_at', () => {
    expect(isHistoricalRun('2023-12-31', '2026-06-16T00:00:00Z')).toBe(true);
  });
  it('false when issue_date is fresh (within 14 days)', () => {
    expect(isHistoricalRun('2026-06-10', '2026-06-16T00:00:00Z')).toBe(false);
  });
  it('false when issueDate missing', () => {
    expect(isHistoricalRun(undefined, '2026-06-16T00:00:00Z')).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `bun run test:unit isHistoricalRun` (หรือ `bunx vitest run services/isHistoricalRun.test.ts`)
Expected: FAIL (`isHistoricalRun is not a function`)

- [ ] **Step 3: Add `isHistoricalRun` to `forecastService.ts`**

เพิ่ม export (วางใกล้ `formatForecastDate`):

```typescript
/** true ถ้า issue_date เก่ากว่า staleDays เทียบ generatedAt (ใช้ตัดสินว่าโชว์ banner historical ไหม) */
export function isHistoricalRun(
  issueDate?: string,
  generatedAt?: string,
  staleDays = 14,
): boolean {
  if (!issueDate) return false;
  const ref = generatedAt ? new Date(generatedAt) : new Date();
  const issued = new Date(issueDate + 'T00:00:00Z');
  const ageDays = (ref.getTime() - issued.getTime()) / 86_400_000;
  return ageDays > staleDays;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `bunx vitest run services/isHistoricalRun.test.ts`
Expected: 3 passed

- [ ] **Step 5: Use it in `HistoricalRunBanner.tsx`**

แก้ signature + guard:
```tsx
import { formatForecastDate, isHistoricalRun } from '@/services/forecastService';

export function HistoricalRunBanner({ issueDate, generatedAt }: { issueDate?: string; generatedAt?: string }) {
  const { isDarkMode, language } = useSettings();
  const theme = Colors[isDarkMode ? 'dark' : 'light'];
  if (!isHistoricalRun(issueDate, generatedAt)) return null;
  const issued = formatForecastDate(issueDate!);
  // ... (ส่วนที่เหลือคงเดิม)
```

- [ ] **Step 6: Pass `generatedAt` in `alerts.tsx`**

แก้บรรทัดที่ render banner:
```tsx
<HistoricalRunBanner issueDate={mapPoints[0]?.issue_date} generatedAt={mapPoints[0]?.generated_at} />
```

- [ ] **Step 7: Run full frontend unit suite**

Run: `bun run test:unit`
Expected: ทุก test ผ่าน (deepseekContract + isHistoricalRun)

- [ ] **Step 8: Commit**

```bash
git add services/forecastService.ts services/isHistoricalRun.test.ts components/forecast/HistoricalRunBanner.tsx app/(tabs)/alerts.tsx
git commit -m "feat: show historical-run banner only when forecast is stale (>14d)"
```

---

## Task 7: Docs — RUNBOOK operational province

**Files:**
- Modify: `docs/RUNBOOK.md`

- [ ] **Step 1: เพิ่มหัวข้อใน RUNBOOK**

เพิ่มท้ายส่วน "C. Pipeline bridge":

```markdown
## D. Operational province forecast (issue_date ปัจจุบัน)
ออก forecast รายจังหวัดจากข้อมูล ERA5 ล่าสุด (ไม่ค้างที่ 2023-12-31):
```
python scripts/freeze_provinces_climatology.py          # ครั้งเดียว: แช่แข็ง thr90 + per-province base_rate
python scripts/download_era5_hourly_aggregate.py recent # ดึง ERA5 ~70 วันล่าสุด -> raw_recent/
python scripts/download_indices.py                      # refresh MJO/Niño34 (จำเป็น ไม่งั้น nino34_lag1m ล่าสุดหาย)
python scripts/predict_provinces.py operational         # ออก forecast issue_date ปัจจุบัน
python scripts/validate_contract.py                     # ตรวจ gate
```
- ERA5 ล่าช้า ~6 วัน -> issue_date จะเป็น ~วันนี้ −6
- MJO หาย -> impute + ใส่ warnings ใน JSON (เหมือน regional)
- การย้าย loop ไปรันอัตโนมัติบน CI = Part 2 (GitHub Actions)
```

- [ ] **Step 2: Commit**

```bash
git add docs/RUNBOOK.md
git commit -m "docs: RUNBOOK operational province forecast steps"
```

---

## Self-Review

**Spec coverage:**
- §4.1 frozen artifact → Task 1 ✓ | §4.2 freeze script → Task 1 ✓ | §4.3 operational build → Task 2 ✓ | §4.4 predict operational → Task 4 ✓ | §4.5 banner conditional → Task 6 ✓
- §3.1 per-province base_rate → Task 1 (freeze) + Task 4 (use) ✓ | §3.4 MJO impute reuse `impute_neutral_mjo` → Task 2 ✓
- §6 error handling: missing artifact/dirs fail loudly → Task 2 `_load_frozen_climatology`/`_soil_grid` raise ✓
- §7 testing: parity → Task 3 ✓ | regime sanity + e2e → Task 4 ✓ | banner → Task 6 ✓ | regression → Task 5 ✓
- §9 ordering matches Tasks 1→7 ✓

**Placeholder scan:** ไม่มี TBD/TODO; ทุก step มีโค้ด/คำสั่ง/expected จริง.

**Type consistency:** `build_provinces_features(verbose, operational)` ใช้ตรงกัน Task 2/3/4; `_load_frozen_climatology()` คืน dict มี `thr90_grid`/`base_rate` ใช้ตรงกัน Task 1/2/4; `isHistoricalRun(issueDate, generatedAt, staleDays)` ตรงกัน Task 6 helper/test/banner; `prov_base[pid][L]` key เป็น int ตรงกับ freeze (`int(L)`).
