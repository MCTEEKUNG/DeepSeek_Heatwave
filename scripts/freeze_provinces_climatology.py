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

    # base_rate มาจาก parquet (groupby province_id) ส่วน pv มาจาก provinces.csv —
    # ต้องตรงกัน ไม่งั้น artifact จะอ้าง n_provinces ผิดเงียบ ๆ (จับ data drift ตอน build)
    assert len(base) == len(pv), f"จำนวนจังหวัดไม่ตรง: parquet={len(base)}, csv={len(pv)}"

    art = {
        "thr90_grid": thr90,
        "base_rate": base,
        "n_provinces": int(len(base)),
        "leads": list(LEADS),
        "window": PCTL_WINDOW,
        "min_run": MIN_RUN,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": str(TMAX_DIR),
    }
    CLIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    # pickle is safe here: artifact is generated and consumed entirely within this
    # codebase from trusted local sources (ERA5 .nc files + project parquet).
    with open(CLIM_FILE, "wb") as fh:
        pickle.dump(art, fh)
    if verbose:
        print(f"[OK] {CLIM_FILE} | thr90 {dict(thr90.sizes)} | base_rate {len(base)} จังหวัด x {len(LEADS)} lead")
    return art


if __name__ == "__main__":
    freeze()
