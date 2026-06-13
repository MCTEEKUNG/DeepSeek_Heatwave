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
    sub = df[list(dict.fromkeys(FEATURES_P + ["doy", "province_id", "in_hw_today", "date", col]))].dropna().sort_values("date")
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
