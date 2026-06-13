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
PER_PROVINCE_MIN_N = 50
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
    _want = list(dict.fromkeys(FEATURES_P + ["doy", "province_id", "in_hw_today", "date", col]))
    sub = df[_want].dropna().sort_values("date")
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
        p_clim, p_pers = _baseline_by_province(tr, te, col)
        block = te[["province_id", "date", col]].copy()
        block["p"] = p
        block["p_clim"] = p_clim
        block["p_pers"] = p_pers
        preds.append(block)
    pred = pd.concat(preds, ignore_index=True).dropna(subset=["p_clim", "p_pers"])
    pooled = evaluate_probabilistic(pred[col].to_numpy(), pred["p"].to_numpy(),
                                    baseline_prob=pred["p_clim"].to_numpy())
    pooled = {"lead": lead, **pooled}
    pooled["bss_vs_persist"] = brier_skill_score(pred[col].to_numpy(), pred["p"].to_numpy(),
                                                 baseline_prob=pred["p_pers"].to_numpy())
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
    pred = pred.rename(columns={col: "y"})
    pred["lead"] = lead
    return pooled, per_prov, pred


def main() -> int:
    df = pd.read_parquet(OUT_FILE)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== train_provinces: {len(df)} แถว | {PROD_MODEL}_cal | leads={LEADS} ===")
    pooled_rows, per_prov_all = [], []
    all_preds = []
    for lead in LEADS:
        pooled, per_prov, pred = run_lead(df, lead)
        pooled_rows.append(pooled)
        per_prov_all.append(per_prov)
        all_preds.append(pred[["province_id", "date", "lead", "y", "p", "p_clim", "p_pers"]])
    pooled_df = pd.DataFrame(pooled_rows)
    per_prov_df = pd.concat(per_prov_all, ignore_index=True)
    pooled_df.to_csv(ANALYSIS_DIR / "provinces_pooled_bss.csv", index=False)
    per_prov_df.to_csv(ANALYSIS_DIR / "provinces_per_province_bss.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(ANALYSIS_DIR / "provinces_predictions.csv", index=False)
    n_win = int((pooled_df["bss"] > 0).sum())
    print(f"\npooled BSS>0: {n_win}/{len(LEADS)} leads")
    print(pooled_df.round(3).to_string(index=False))
    print(f"[OK] ผลที่ {ANALYSIS_DIR}/provinces_*.csv")
    return 0


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
