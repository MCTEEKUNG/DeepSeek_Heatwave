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
                   fit_predict_calibrated)
from train_final import PROD_MODEL
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
    deltas = []
    for lead in LEADS:
        b = pooled_bss(df, lead, leakfree=False)
        f = pooled_bss(df, lead, leakfree=True)
        d = f - b
        deltas.append(d)
        print(f"{lead:>4} | {b:>+10.4f} | {f:>+12.4f} | {d:>+8.4f} | "
              f"{'เล็ก' if abs(d) < SCREEN_DBSS else 'ตรวจเทียบ CI'}")
    # NaN-aware: ถ้า lead ใดคืน nan (เช่น fold ถูกข้ามหมด) ให้ worst เป็น nan ไม่ใช่ 0 ปลอม
    worst = float(np.nanmax(np.abs(deltas))) if not all(np.isnan(deltas)) else float("nan")
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
        sys.exit(main())
