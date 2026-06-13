# Rigor + Serving Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** กำจัด leakage 2 ตัว (in_hw_today มองอนาคต + percentile-label leak) ให้สกิลที่รายงานป้องกันได้ แล้วยืนยัน serving path ออก forecast ได้จริง — ทั้งหมดบน pipeline เดียว (train/serve parity)

**Architecture:** R2 = แก้ `in_hw_today` ที่ต้นทาง (`build_dataset.lookback_features`) ให้เป็น trailing-only โดยใช้ helper ใหม่ใน `heatwave_target.py` → กระทบทั้ง feature โมเดลและ persistence baseline พร้อมกัน. R1 = วัด ΔBSS แบบ leave-block-out หนึ่งครั้ง (gate) ตัดสินว่าต้อง refactor ไหม. Serving = retrain final บน dataset ที่แก้แล้ว + กระชับ parity self-test + runbook (สคริปต์ดึงข้อมูลล่าสุด `recent` มีอยู่แล้ว).

**Tech Stack:** Python 3.12, xarray, pandas, numpy, scikit-learn, lightgbm, cdsapi ; test แบบ inline `_selftest()` รันผ่าน `python scripts/<module>.py [test]` (ธรรมเนียมของ repo นี้ — ไม่ใช่ pytest)

**สเปกอ้างอิง:** `docs/superpowers/specs/2026-06-13-rigor-serving-hardening-design.md`

---

## Prerequisites

- branch `feat/harden-rigor-serving` ถูกสร้างและ checkout แล้ว (spec commit อยู่บนนี้)
- ติดตั้ง deps: `python -m pip install -r requirements.txt`
- console ภาษาไทยบน Windows: `set PYTHONIOENCODING=utf-8` (cmd) หรือ `$env:PYTHONIOENCODING="utf-8"` (PowerShell) ก่อนรันสคริปต์
- ข้อมูล raw ครบแล้ว (Tmax 60 ไฟล์, soil 120 ไฟล์, indices) — **ไม่ต้องดาวน์โหลดใหม่**

**Deviation จาก spec:** spec §6 ระบุ S1 = เขียน `scripts/download_recent.py` ใหม่ — แต่ฟังก์ชันนี้**มีอยู่แล้ว**ใน `scripts/download_era5_hourly_aggregate.py` (โหมด `recent`, บรรทัด 117–181). แผนนี้จึงเปลี่ยน S1 เป็น "ใช้/บันทึก runbook ของโหมดที่มีอยู่" แทนการเขียนใหม่

---

## Phase 1 — Rigor (แก้ leak → re-validate → gate)

### Task 1: helper `trailing_run_length` (R2 ส่วนตรรกะ)

**Files:**
- Modify: `scripts/heatwave_target.py` (เพิ่มฟังก์ชัน + unit test ใน `__main__`)

- [ ] **Step 1: เขียน failing test** — เพิ่มบล็อกนี้ใน `scripts/heatwave_target.py` ใต้ assert ของ `flag_heatwaves` (หลังบรรทัด `print("  [OK] ตรรกะถูกต้อง\n")`)

```python
    # --- unit test: trailing run length (มองย้อนหลังเท่านั้น = leak-free) ---
    seq2 = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
    exp_run = np.array([0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4])
    got_run = trailing_run_length(seq2)
    assert (got_run == exp_run).all(), got_run.tolist()
    # in_hw (trailing >= 3): ติด 1 ตั้งแต่ "วันที่ 3" ของ streak เป็นต้นไป (ไม่ย้อนติดให้ 2 วันแรก)
    assert ((exp_run >= 3).astype(int).tolist()
            == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1])
    print("  [OK] trailing_run_length + in_hw(trailing>=3) ถูกต้อง\n")
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `python scripts/heatwave_target.py`
Expected: FAIL — `NameError: name 'trailing_run_length' is not defined`

- [ ] **Step 3: เขียน implementation** — เพิ่มฟังก์ชันนี้ใน `scripts/heatwave_target.py` ใต้ `flag_heatwaves` (ก่อน `if __name__`)

```python
def trailing_run_length(hot_bool) -> np.ndarray:
    """ความยาว run ของ "วันร้อน" ที่ต่อเนื่องและ **จบ ณ ตำแหน่งนั้น** (มองเฉพาะ index <= t).

    รับ 1D (array/Series ของ 0/1/NaN) คืน np.ndarray ความยาวเท่ากัน.
    ค่า != 1 (รวม 0 และ NaN) = ตัด streak (รีเซ็ตเป็น 0).
    leak-free โดยนิยาม: ไม่มองอนาคต — ต่างจาก flag_heatwaves ที่นับ fwd+bwd (ใช้ทำ label).
    """
    arr = np.asarray(hot_bool, dtype=float)
    out = np.zeros(arr.shape[0], dtype=float)
    run = 0.0
    for i in range(arr.shape[0]):
        run = run + 1.0 if arr[i] == 1.0 else 0.0
        out[i] = run
    return out
```

- [ ] **Step 4: รัน test ให้ผ่าน**

Run: `python scripts/heatwave_target.py`
Expected: PASS — เห็น `[OK] trailing_run_length + in_hw(trailing>=3) ถูกต้อง`

- [ ] **Step 5: commit**

```bash
git add scripts/heatwave_target.py
git commit -m "$(printf 'feat: add trailing_run_length (leak-free hot-streak)\n\nBackward-only run length ending at each day, for the in_hw_today\nfeature fix (R2). flag_heatwaves (fwd+bwd) stays for labels.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: wire `in_hw_today` → trailing + leakage self-test + rebuild (R2)

**Files:**
- Modify: `scripts/build_dataset.py` (import, `lookback_features`, `_selftest`)

- [ ] **Step 0: snapshot ตัวเลข "ก่อนแก้" สำหรับ before/after** (outputs/ ถูก gitignore — สำเนาไว้เทียบ local)

Run: `python -c "import shutil,os; os.makedirs('outputs/analysis',exist_ok=True); shutil.copy('outputs/metrics_pooled.csv','outputs/metrics_pooled_BEFORE.csv'); print('snapshot ok')"`
Expected: `snapshot ok` (ถ้าไม่มีไฟล์เดิม ข้าม step นี้ — จะได้ตัวเลข before จาก git history/results_master.md แทน)

- [ ] **Step 1: เขียน failing test** — เพิ่มบล็อกนี้ใน `_selftest()` ของ `scripts/build_dataset.py` ก่อนบรรทัด `print("[OK] self-test ผ่านทั้งหมด")`

```python
    # 6) in_hw_today (trailing) มองย้อนหลังเท่านั้น + ติด 1 ที่ "วันที่ 3" ของ streak
    idxc = pd.date_range("2020-01-01", periods=20, freq="D")
    hot = pd.Series(0.0, index=idxc)
    hot.iloc[5:8] = 1.0  # ร้อน 3 วันติด (index 5,6,7)
    daily_c = pd.DataFrame({"sm1": 0.30, "sm3": 0.35, "tmax_rm": 30.0,
                            "hot_rm": hot, "hw_rm": 0.0}, index=idxc)
    fa = lookback_features(daily_c)
    inhw_a = fa["in_hw_today"].to_numpy()
    assert inhw_a[7] == 1.0 and inhw_a[6] == 0.0 and inhw_a[5] == 0.0, inhw_a[:9].tolist()
    # เปลี่ยน "อนาคต" (index 8,9) ให้ร้อนต่อ -> in_hw ของ index <= 7 ต้องไม่ขยับ (leak-free)
    daily_d = daily_c.copy()
    daily_d.loc[idxc[8:10], "hot_rm"] = 1.0
    fb = lookback_features(daily_d)
    assert np.array_equal(inhw_a[:8], fb["in_hw_today"].to_numpy()[:8]), \
        "in_hw_today leak: ค่าอนาคตกระทบอดีต"
    print("[OK] in_hw_today trailing-only: ติด 1 ที่วันที่ 3 ของ streak, อนาคตไม่กระทบอดีต")
```

- [ ] **Step 2: รัน test ให้เห็นว่า fail**

Run: `python scripts/build_dataset.py test`
Expected: FAIL — `AssertionError` ที่ `inhw_a[7] == 1.0` (โค้ดเดิมตั้ง `in_hw_today = hw_rm` ซึ่ง = 0 ทั้งหมดใน daily_c)

- [ ] **Step 3: implement — แก้ import** ที่หัวไฟล์ `scripts/build_dataset.py`

เปลี่ยน:
```python
from heatwave_target import (
    load_tmax_celsius,
    doy_window_percentile,
    hot_days,
    flag_heatwaves,
)
```
เป็น:
```python
from heatwave_target import (
    load_tmax_celsius,
    doy_window_percentile,
    hot_days,
    flag_heatwaves,
    trailing_run_length,
)
```

- [ ] **Step 4: implement — แก้ `lookback_features`** ใน `scripts/build_dataset.py`

เปลี่ยน:
```python
    f["in_hw_today"] = d["hw_rm"]
    f["hot_frac7"] = d["hot_rm"].rolling(7, min_periods=7).mean()
```
เป็น:
```python
    # in_hw_today (trailing-only, leak-free): อยู่ใน hot-streak ติดต่อกัน >= MIN_RUN ที่จบ ณ วันนี้
    # ใช้ hot_rm (same-day, ไม่มองอนาคต) ไม่ใช่ hw_rm (fwd+bwd = ใช้ทำ label เท่านั้น)
    streak = trailing_run_length(d["hot_rm"].to_numpy())
    in_hw = pd.Series(streak >= MIN_RUN, index=full).astype(float)
    in_hw[d["hot_rm"].isna()] = np.nan      # วันไม่มีข้อมูล -> NaN (สอดคล้องกับ feature อื่น)
    f["in_hw_today"] = in_hw
    f["hot_frac7"] = d["hot_rm"].rolling(7, min_periods=7).mean()
```

- [ ] **Step 5: รัน self-test ทั้งไฟล์ให้ผ่าน**

Run: `python scripts/build_dataset.py test`
Expected: PASS — เห็นทั้ง `[OK] features มองย้อนหลังเท่านั้น...` (test เดิม #3) และ `[OK] in_hw_today trailing-only...` (test ใหม่ #6) และ `[OK] self-test ผ่านทั้งหมด`

- [ ] **Step 6: rebuild dataset (เต็ม) — สร้าง dataset.csv + climatology.pkl ใหม่**

Run: `python scripts/build_dataset.py`
Expected: เห็น `[OK] .../dataset.csv` + บรรทัด base rate ราย lead (rows=10957) ; `models/climatology.pkl` ถูกเขียนใหม่

- [ ] **Step 7: commit** (dataset.csv อยู่ใต้ data/ = gitignore ไม่ commit ; climatology.pkl ใต้ models/ = tracked)

```bash
git add scripts/build_dataset.py models/climatology.pkl
git commit -m "$(printf 'fix: in_hw_today is trailing-only, not forward-looking (R2)\n\nflag_heatwaves uses fwd+bwd run counting -- correct for the LABEL but\na leak when its output (hw_rm) was used as the in_hw_today FEATURE\n(uses t+1,t+2 to flag day t; also breaks serve consistency at the\nlatest day). Rebuild in_hw_today from trailing hot-streak of hot_rm.\nFixed at source so the persistence baseline (train.py state) picks up\nthe same definition.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: re-validate (rerun train + analysis) + integrity note

**Files:**
- Create: `docs/INTEGRITY.md` (บันทึก R1/R2 + before/after)

- [ ] **Step 1: รัน train (CV + baselines + calibration)**

Run: `python scripts/train.py`
Expected: ตาราง `BSS (pooled...) — y_rm` ; โมเดลหลักส่วนใหญ่ BSS > 0 ; เขียน `outputs/metrics_pooled.csv`, `metrics_folds.csv`, `predictions.csv`, figures

- [ ] **Step 2: รันชุด analysis (robustness + รายงาน)**

Run: `python scripts/analysis/make_report.py`
Expected: อัปเดต `outputs/analysis/results_master.md` (+ bootstrap_ci, ablation, calibration ฯลฯ) โดยไม่ error
หมายเหตุ: ถ้า `make_report.py` ต้องการ args/ลำดับ ให้ดู docstring หัวไฟล์ก่อนรัน

- [ ] **Step 3: เทียบ before/after แล้วบันทึก** — สร้าง `docs/INTEGRITY.md` ด้วยเนื้อหานี้ (เติมตัวเลขจริงจาก `metrics_pooled.csv` ปัจจุบัน เทียบ `metrics_pooled_BEFORE.csv` / git history)

```markdown
# Integrity Note — leakage audit (2026-06-13)

ก่อน "ลงหลัก" ที่โปรเจกต์นี้ มีการตรวจและแก้ leakage 2 จุด ดูดีไซน์เต็มที่
`docs/superpowers/specs/2026-06-13-rigor-serving-hardening-design.md`.

## R2 — in_hw_today มองอนาคต (แก้แล้ว: eliminate)
`flag_heatwaves` นับ run แบบ fwd+bwd (ถูกต้องสำหรับ label) แต่ output เดิมถูกใช้เป็น
feature `in_hw_today` และ state ของ persistence baseline → ใช้ข้อมูลอนาคต ณ วันออกพยากรณ์
และทำให้ค่าวันล่าสุดตอน serve ไม่ตรง. แก้เป็น trailing-only (`trailing_run_length`).

ผลต่อสกิล (pooled BSS, y_rm, prod model logistic_balanced_cal):

| lead | BSS ก่อน (leak) | BSS หลัง (trailing) | beats clim | beats persist |
| --- | --- | --- | --- | --- |
| 2 | <เติม> | <เติม> | <✓/·> | <✓/·> |
| 3 | <เติม> | <เติม> | <✓/·> | <✓/·> |
| 4 | <เติม> | <เติม> | <✓/·> | <✓/·> |
| 5 | <เติม> | <เติม> | <✓/·> | <✓/·> |
| 6 | <เติม> | <เติม> | <✓/·> | <✓/·> |

> persistence baseline ขยับเพราะมันพึ่ง in_hw_today — "beats persistence" รอบนี้เทียบของสะอาดทั้งคู่ (คาดได้ ไม่ใช่ regression)

## R1 — percentile-label leak (gate: ดู Task 4)
เกณฑ์ p90 ที่นิยาม label คำนวณบน 30 ปีเต็มรวม test. วัด ΔBSS แบบ leave-block-out:
ΔBSS ที่วัดได้ = <เติมจาก leak_check_r1.py> → <อยู่ใน / หลุด> 95% CI ของ results_master
→ การตัดสิน: <document พอ (คง frozen-all-history) / ทำ per-fold elimination>
```

- [ ] **Step 4: commit** (outputs/ gitignore — commit เฉพาะ docs)

```bash
git add docs/INTEGRITY.md
git commit -m "$(printf 'docs: integrity note + before/after BSS after R2 fix\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: R1 leak gate — วัด ΔBSS leave-block-out (วัดครั้งเดียว)

**Files:**
- Create: `scripts/analysis/leak_check_r1.py`

- [ ] **Step 1: เขียน failing self-test** — สร้าง `scripts/analysis/leak_check_r1.py` ด้วยเนื้อหาเต็มนี้

```python
"""R1 leak gate (วัดครั้งเดียว ตาม spec 2026-06-13).

วัดผลกระทบของ percentile-label leak ต่อ pooled BSS:
  baked  = label ที่เกณฑ์ p90 fit จาก 30 ปีเต็ม (ติด leak — ของจริงใน dataset.csv)
  leakfree = label ที่เกณฑ์ p90 fit จาก "วันใน train-fold เท่านั้น" ต่อ fold
ถ้า |ΔBSS| เล็กกว่าครึ่งหนึ่งของ 95% CI ใน results_master -> leak ไม่ significant -> document พอ.

ใช้งาน:  python scripts/analysis/leak_check_r1.py        # รันบน dataset.csv จริง
         python scripts/analysis/leak_check_r1.py test   # self-test: relabel เต็ม = baked
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from heatwave_target import doy_window_percentile, hot_days, flag_heatwaves
from build_dataset import weekly_event_targets, MIN_RUN, PCTL_WINDOW
from cv import RollingOriginCV
from train import (FEATURES, LEADS, PRIMARY_TARGET, GAP, N_SPLITS, TEST_SIZE,
                   make_estimator, fit_predict_calibrated, PROD_MODEL)
from evaluate import predict_seasonal_climatology, brier_skill_score

DATASET = ROOT / "data" / "processed" / "dataset.csv"
# heuristic คัดกรอง: ถ้า |ΔBSS| < นี้ ถือว่าเล็กแน่ (CI half-width ใน results_master ~0.05-0.09)
SCREEN_DBSS = 0.02


def _tmax_dataarray(df: pd.DataFrame) -> xr.DataArray:
    """อนุกรม regional-mean Tmax รายวันจาก dataset.csv -> DataArray dim 'time'."""
    return xr.DataArray(df["tmax_rm"].to_numpy(dtype=float), dims="time",
                        coords={"time": df.index.values})


def weekly_labels_from_threshold(da: xr.DataArray, fit_da: xr.DataArray,
                                 lead: int, q: float = 90) -> pd.Series:
    """label รายสัปดาห์ของ lead นี้ โดย fit เกณฑ์ p90 จาก fit_da แล้ว apply กับ da เต็ม."""
    thr = doy_window_percentile(fit_da, q=q, window=PCTL_WINDOW)
    hw = flag_heatwaves(hot_days(da, thr), min_len=MIN_RUN)
    flag = pd.Series(hw.values.astype(float), index=pd.DatetimeIndex(da["time"].values))
    return weekly_event_targets(flag, [lead])[f"lead{lead}"]


def pooled_bss(df: pd.DataFrame, lead: int, leakfree: bool) -> float:
    """pooled BSS ของ PROD model ที่ lead นี้ ; leakfree=True -> relabel ต่อ fold (train-only threshold)."""
    da = _tmax_dataarray(df)
    col = f"{PRIMARY_TARGET}_l{lead}"
    sub = df[FEATURES + ["doy", col]].dropna().sort_index()
    X = sub[FEATURES].to_numpy(dtype=float)
    doy = sub["doy"].to_numpy(dtype=int)
    dates = sub.index
    y_baked = sub[col].to_numpy(dtype=float)

    cv = RollingOriginCV(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=GAP, expanding=True)
    ys, ps, pcs = [], [], []
    for tr, te in cv.split(len(sub)):
        if leakfree:
            fit_da = da.sel(time=slice(None, np.datetime64(dates[tr].max())))
            y_full = weekly_labels_from_threshold(da, fit_da, lead).reindex(dates)
            y = y_full.to_numpy(dtype=float)
            keep_tr = tr[~np.isnan(y[tr])]
            keep_te = te[~np.isnan(y[te])]
        else:
            y = y_baked
            keep_tr, keep_te = tr, te
        if len(np.unique(y[keep_tr])) < 2 or len(keep_te) == 0:
            continue
        p = fit_predict_calibrated(PROD_MODEL, X[keep_tr], y[keep_tr], X[keep_te])
        if p is None:
            continue
        p_clim = predict_seasonal_climatology(doy[keep_tr], y[keep_tr], doy[keep_te])
        ys.append(y[keep_te]); ps.append(p); pcs.append(p_clim)
    if not ys:
        return float("nan")
    y_all = np.concatenate(ys); p_all = np.concatenate(ps); pc_all = np.concatenate(pcs)
    return brier_skill_score(y_all, p_all, baseline_prob=pc_all)


def main() -> int:
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date").sort_index()
    print(f"=== R1 leak gate: {PRIMARY_TARGET} | {PROD_MODEL}_cal | {len(df)} วัน ===")
    print(f"{'lead':>4} | {'BSS baked':>10} | {'BSS leakfree':>12} | {'dBSS':>8} | verdict")
    worst = 0.0
    for lead in LEADS:
        b = pooled_bss(df, lead, leakfree=False)
        f = pooled_bss(df, lead, leakfree=True)
        d = f - b
        worst = max(worst, abs(d))
        print(f"{lead:>4} | {b:>+10.4f} | {f:>+12.4f} | {d:>+8.4f} | "
              f"{'เล็ก' if abs(d) < SCREEN_DBSS else 'ตรวจเทียบ CI'}")
    print(f"\nΔBSS สูงสุด = {worst:.4f}")
    print(f"เกณฑ์ตัดสิน: เทียบ |ΔBSS| กับครึ่งหนึ่งของ 95% CI ของ BSS ใน "
          f"outputs/analysis/results_master.md")
    print("  - อยู่ใน CI -> document พอ, คง frozen-all-history labels (ไม่ refactor)")
    print("  - หลุด CI  -> ทำ per-fold elimination (ดู spec §4 R1)")
    return 0


def _selftest() -> None:
    """relabel ด้วยเกณฑ์ที่ fit จาก 'ข้อมูลเต็ม' ต้อง reproduce baked y_rm (พิสูจน์ reuse ถูกต้อง)."""
    if not DATASET.exists():
        print("[ข้าม] ไม่มี dataset.csv — รัน build_dataset.py ก่อน")
        return
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date").sort_index()
    da = _tmax_dataarray(df)
    lead = LEADS[0]
    y_recompute = weekly_labels_from_threshold(da, da, lead).reindex(df.index)
    baked = df[f"{PRIMARY_TARGET}_l{lead}"]
    both = (~y_recompute.isna()) & (~baked.isna())
    agree = float((y_recompute[both] == baked[both]).mean())
    assert agree > 0.999, f"relabel(full) ควร = baked แต่ตรงกัน {agree:.4f}"
    print(f"[OK] relabel ด้วยเกณฑ์เต็ม = baked y_rm ({agree*100:.2f}% ตรง, n={int(both.sum())})")
    print("[OK] self-test ผ่าน")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        main()
```

- [ ] **Step 2: รัน self-test ให้เห็นว่าผ่าน (reuse logic ถูกต้อง)**

Run: `python scripts/analysis/leak_check_r1.py test`
Expected: PASS — `[OK] relabel ด้วยเกณฑ์เต็ม = baked y_rm (~100% ตรง...)` (พิสูจน์ว่าการ recompute label ตรงกับ build_dataset)

- [ ] **Step 3: รัน gate จริง บน dataset.csv**

Run: `python scripts/analysis/leak_check_r1.py`
Expected: ตาราง dBSS ราย lead + `ΔBSS สูงสุด = ...` ; คาดว่า |ΔBSS| < 0.02 ทุก lead (leak หักล้างใน BSS ratio ตาม spec §4)

- [ ] **Step 4: บันทึกผลลง `docs/INTEGRITY.md`** — เติมตัวเลข ΔBSS + verdict ในส่วน "R1" (จาก Task 3 Step 3) ตามผลที่ได้

- [ ] **Step 5: commit**

```bash
git add scripts/analysis/leak_check_r1.py docs/INTEGRITY.md
git commit -m "$(printf 'feat: R1 leak gate -- measure label-threshold leak (dBSS)\n\nLeave-block-out relabel vs baked labels; reuses build_dataset +\nheatwave_target so the recompute is faithful (self-test: full-fit\nrelabel reproduces baked y_rm). Verdict recorded in INTEGRITY.md.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

> **Gate decision:** ถ้า ΔBSS หลุด CI → หยุดที่นี่ แล้วเพิ่ม task ใหม่ทำ per-fold elimination ใน `train.py`/`train_final.py` ก่อนไป Phase 2. ถ้าอยู่ใน CI (คาด) → ไป Phase 2 ได้เลย.

---

## Phase 2 — Serving (ทับบนโมเดลที่แก้ leak แล้ว)

### Task 5: retrain final models (S2)

**Files:**
- Modify (regenerate): `models/heatwave_y_rm_lead{2,3,4,5,6}.pkl` (tracked)

- [ ] **Step 1: ตรวจ self-test ของ train_final ก่อน**

Run: `python scripts/train_final.py test`
Expected: PASS — `[OK] artifact ทำนายได้ + ตรงกับ fit_calibrated_model ...`

- [ ] **Step 2: retrain บน dataset ที่แก้ R2 แล้ว**

Run: `python scripts/train_final.py`
Expected: 5 บรรทัด `lead L: n=..., base_rate=..., train YYYY-MM-DD..YYYY-MM-DD` + `[OK] เซฟ 5 โมเดล`

- [ ] **Step 3: ยืนยัน train_issue_doy ครอบเต็มปี** (เลิกเตือน out-of-domain ครึ่งปีหลัง)

Run: `python -c "import pickle; a=pickle.load(open('models/heatwave_y_rm_lead2.pkl','rb')); print('issue_doy', a['train_issue_doy_min'], '-', a['train_issue_doy_max'], '| train', a['train_start'], a['train_end'])"`
Expected: `train_issue_doy_max` ใกล้ 365 (ไม่ใช่ ~212) ; train_end ใกล้ 2023-12

- [ ] **Step 4: commit** (models/ tracked)

```bash
git add models/heatwave_y_rm_lead2.pkl models/heatwave_y_rm_lead3.pkl models/heatwave_y_rm_lead4.pkl models/heatwave_y_rm_lead5.pkl models/heatwave_y_rm_lead6.pkl
git commit -m "$(printf 'feat: retrain final models on leak-fixed full-year dataset (S2)\n\nin_hw_today now trailing-only; train_issue_doy now covers the full\nyear so predict no longer warns out-of-domain in H2.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 6: serving end-to-end + กระชับ parity self-test (S3)

**Files:**
- Modify: `scripts/predict.py` (`_selftest` — เอา 7-day exclusion ออก)
- Modify (regenerate): `docs/forecast.json` (tracked)

- [ ] **Step 1: กระชับ parity test** — ใน `scripts/predict.py._selftest()` operational parity block

เปลี่ยน:
```python
        # เทียบเฉพาะวัน "ภายใน" (ตัด 7 วันท้าย = ขอบ window ที่ in_hw_today ต่างได้)
        idx_in = feat_op.dropna(subset=FEATURES).index[:-7]
```
เป็น:
```python
        # R2 แก้แล้ว: in_hw_today เป็น trailing-only -> วันล่าสุด serve-consistent
        # ไม่ต้องตัด 7 วันท้ายอีก (เดิมตัดเพราะ in_hw_today มองอนาคตที่ขอบ window)
        idx_in = feat_op.dropna(subset=FEATURES).index
```

- [ ] **Step 2: รัน parity self-test ให้ผ่าน (หลักฐานว่า R2 หาย)**

Run: `python scripts/predict.py test`
Expected: PASS — `[OK] operational parity ... วันภายในตรง dataset เป๊ะ` **โดยไม่ตัด 7 วันท้าย** + `[OK] self-test ผ่านทั้งหมด`
(ถ้า fail = ยังมี feature ที่ไม่ serve-consistent — ต้องสอบสวนก่อน commit)

- [ ] **Step 3: รัน predict (full data) ออก forecast.json**

Run: `python scripts/predict.py`
Expected: บรรทัด `lead L สัปดาห์ (...): p=... (...x ปกติ) -> ...` ครบ 5 lead โดย **ไม่มี ⚠️ นอกโดเมน** ; เขียน `docs/forecast.json`

- [ ] **Step 4: ยืนยัน forecast.json in-domain ทุก lead**

Run: `python -c "import json; d=json.load(open('docs/forecast.json',encoding='utf-8')); print('issue', d['issue_date']); print('in_domain', [f['in_training_domain'] for f in d['forecasts']])"`
Expected: `in_domain [True, True, True, True, True]`

- [ ] **Step 5: commit**

```bash
git add scripts/predict.py docs/forecast.json
git commit -m "$(printf 'fix: tighten predict parity test (no 7-day cut) + refresh forecast (S3)\n\nin_hw_today is trailing-only now, so the latest-day feature is\nserve-consistent; operational parity holds to the series end.\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 7: runbook + README + ปิด integrity note (S4)

**Files:**
- Create: `docs/RUNBOOK.md`
- Modify: `README.md` (อัปเดตสถานะ)

- [ ] **Step 1: เขียน `docs/RUNBOOK.md`** ด้วยเนื้อหานี้

```markdown
# RUNBOOK — ออก forecast ปัจจุบันด้วยมือ (on-demand)

## A. Forecast จากข้อมูลเทรนเต็ม (ไม่ต้องมี CDS key)
ออกพยากรณ์จาก "วันล่าสุดที่ feature ครบ" ในข้อมูลเทรน (ใช้ตรวจ/เดโม):
```
python scripts/predict.py
```
ผล -> `docs/forecast.json`

## B. Forecast "ปัจจุบันจริง" (ต้องมี CDS key ที่ ~/.cdsapirc)
1. ดึงข้อมูล ERA5 ล่าสุด ~70 วัน เข้า data/raw_recent/ (โหมด recent มีอยู่แล้ว):
   ```
   python scripts/download_era5_hourly_aggregate.py recent
   ```
2. รีเฟรช indices ล่าสุด (MJO/Niño):
   ```
   python scripts/download_indices.py
   ```
3. ออกพยากรณ์แบบ operational (climatology แช่แข็ง + raw_recent/):
   ```
   python scripts/predict.py operational
   ```
ผล -> `docs/forecast.json` (issue date = วันล่าสุดที่ feature ครบ ; MJO ที่ค้างถูก impute + ใส่ warning)

## หมายเหตุ
- ERA5 ล่าช้า ~5-6 วัน (ตั้งไว้ใน ERA5_LATENCY_DAYS) -> issue date จะตามหลังวันนี้เล็กน้อย
- ถ้า retrain โมเดล/เปลี่ยนนิยาม feature: รัน build_dataset.py -> train_final.py -> predict.py ตามลำดับ
- ทุกโมดูลมี self-test: `python scripts/<module>.py test`
```

- [ ] **Step 2: อัปเดต README** — แก้บล็อก "## สถานะ" ใน `README.md` เพิ่มบรรทัดใต้รายการเดิม

```markdown

## สถานะ (2026-06-13) — leakage audit + serving hardening ✅

- [x] R2: `in_hw_today` เป็น trailing-only (กัน forward-looking feature + serve-consistent)
- [x] R1: วัด ΔBSS percentile-label leak (gate) — ดู `docs/INTEGRITY.md`
- [x] retrain final ครอบเต็มปี (issue doy เต็ม) + parity self-test ไม่ต้องตัด 7 วันท้าย
- [x] runbook ออก forecast on-demand: `docs/RUNBOOK.md`
- รายละเอียด: `docs/superpowers/specs/2026-06-13-rigor-serving-hardening-design.md`
```

- [ ] **Step 3: ปิด integrity note** — ยืนยันว่า `docs/INTEGRITY.md` เติมตัวเลข before/after (Task 3) + ΔBSS verdict (Task 4) ครบ ไม่มี `<เติม>` ค้าง

Run: `grep -n "เติม" docs/INTEGRITY.md || echo "no placeholders"`
Expected: `no placeholders`

- [ ] **Step 4: commit**

```bash
git add docs/RUNBOOK.md README.md docs/INTEGRITY.md
git commit -m "$(printf 'docs: runbook for on-demand forecast + README status (S4)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Acceptance criteria (ตรวจปิดงาน)

1. `python scripts/heatwave_target.py` + `python scripts/build_dataset.py test` ผ่าน (รวม test in_hw_today trailing-only)
2. `python scripts/predict.py test` ผ่าน **โดยไม่ตัด 7 วันท้าย**
3. `docs/INTEGRITY.md`: ตาราง BSS ก่อน/หลัง R2 + ΔBSS R1 + verdict ครบ ไม่มี placeholder
4. `docs/forecast.json`: `in_training_domain = true` ทุก lead
5. `docs/RUNBOOK.md` + README สถานะใหม่ commit แล้ว
6. self-test ทุกโมดูลผ่าน: `heatwave_target, build_dataset, cv, train, train_final, predict, evaluate, leak_check_r1`

---

## Self-review (ผู้เขียนแผนตรวจเองแล้ว)

- **Spec coverage:** R2 → Task 1–2 ; R1 gate → Task 4 ; re-validation/integrity → Task 3 ; S2 retrain → Task 5 ; S3 smoke+parity → Task 6 ; S4 runbook/docs → Task 7. (S1 download_recent = มีอยู่แล้ว → runbook Task 7B)
- **Placeholder scan:** โค้ดทุก step เต็ม ; `<เติม>` ใน INTEGRITY.md เป็นช่องตัวเลขผลลัพธ์ที่ Task 3/4 เติม + มี grep gate (Task 7 Step 3) — ไม่ใช่ placeholder ของโค้ด
- **Type consistency:** `trailing_run_length` (heatwave_target) ใช้ชื่อเดียวกันใน import + lookback_features + leak_check ; `weekly_event_targets`, `MIN_RUN`, `PCTL_WINDOW`, `PROD_MODEL`, `fit_predict_calibrated`, `predict_seasonal_climatology`, `brier_skill_score` ทั้งหมดมีอยู่จริง (ตรวจกับ build_dataset.py / train.py / evaluate.py แล้ว)
- **ลำดับพึ่งพา:** T1→T2 (helper ก่อนใช้) ; rebuild (T2) ก่อน train/gate (T3/T4) ; แก้ leak (P1) ก่อน retrain/serve (P2) — ตรงตาม spec §8
