"""
ตัวช่วยกลางสำหรับชุดวิเคราะห์ robustness (scripts/analysis/)

- โหลด outputs/predictions.csv = แหล่งข้อมูลหลักที่ "อ่านอย่างเดียว"
  (คอลัมน์: target, lead, fold, model, date, y, p, p_clim ; 8 โมเดล ; fold 1-5)
- จัด ENSO regime ต่อ "วันออกพยากรณ์" จาก nino34.csv
- ค่าคงที่กลาง: โมเดลที่ใช้รายงาน, baseline, ชื่อสวยของโมเดล, path ผลลัพธ์

กติกาสำคัญ (ตามแผน):
- พระเอกที่ใช้รายงาน = logistic / lgbm_cal / logistic_balanced_cal
  ห้ามใช้ "lgbm ดิบ" เป็นพระเอกของ y_rm (BSS ติดลบ) — เก็บไว้เล่าเรื่อง calibration เท่านั้น
- ไม่พึ่ง scipy/statsmodels ; ฟังก์ชันสถิติอยู่ที่ stats.py (numpy ล้วน)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ให้ import โมดูลใน scripts/ ได้ (evaluate, cv, train, ...) — analysis อยู่ลึกลงไป 1 ชั้น
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

ROOT = SCRIPTS_DIR.parent
OUT_DIR = ROOT / "outputs"
ANALYSIS_DIR = OUT_DIR / "analysis"
FIG_DIR = ANALYSIS_DIR / "figures"
PRED_FILE = OUT_DIR / "predictions.csv"
POOLED_FILE = OUT_DIR / "metrics_pooled.csv"
NINO_FILE = ROOT / "data" / "processed" / "indices" / "nino34.csv"

PRED_COLUMNS = ["target", "lead", "fold", "model", "date", "y", "p", "p_clim"]

PRIMARY_TARGET = "y_rm"
TARGETS = ["y_rm", "y_rm95", "y_af"]
LEADS = [2, 3, 4, 5, 6]

BASELINES = ["climatology", "persistence"]
# โมเดลที่ "calibrate ดี" และใช้รายงานจริง (ดูกติกาด้านบน)
REPORT_MODELS = ["logistic", "lgbm_cal", "logistic_balanced_cal"]

MODEL_LABELS = {
    "climatology": "Climatology",
    "persistence": "Persistence",
    "logistic": "Logistic",
    "lgbm": "LightGBM (raw)",
    "lgbm_cal": "LightGBM (Platt)",
    "logistic_balanced_cal": "Logistic (bal+cal)",
    "lgbm_balanced_cal": "LightGBM (bal+cal)",
    "balanced_rf_cal": "BalancedRF (cal)",
}

ENSO_THRESH = 0.5  # °C ; |anomaly| >= 0.5 -> El Niño / La Niña (โดยประมาณแบบ ONI ±0.5)


def ensure_dirs() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- predictions

def load_predictions() -> pd.DataFrame:
    """โหลด predictions รายวัน เรียงให้พร้อมใช้ (target, lead, model, fold, date)."""
    df = pd.read_csv(PRED_FILE, parse_dates=["date"])
    df = df.sort_values(["target", "lead", "model", "fold", "date"]).reset_index(drop=True)
    return df


def load_pooled() -> pd.DataFrame:
    return pd.read_csv(POOLED_FILE)


def model_series(pred: pd.DataFrame, target: str, lead: int, model: str) -> pd.DataFrame:
    """แถวของ (target, lead, model) เรียงตาม (fold, date) — หน่วยที่ใช้ทำ block bootstrap."""
    g = pred[(pred["target"] == target) & (pred["lead"] == lead) & (pred["model"] == model)]
    return g.sort_values(["fold", "date"]).reset_index(drop=True)


def iter_combos(pred: pd.DataFrame, targets=None, leads=None, models=None):
    """วนทุก (target, lead, model) ที่มีในข้อมูล (กรองได้) — yield (t, l, m, sub_df เรียงแล้ว)."""
    sub = pred
    if targets is not None:
        sub = sub[sub["target"].isin(targets)]
    if leads is not None:
        sub = sub[sub["lead"].isin(leads)]
    if models is not None:
        sub = sub[sub["model"].isin(models)]
    for (t, l, m), g in sub.groupby(["target", "lead", "model"], sort=True):
        yield t, int(l), m, g.sort_values(["fold", "date"]).reset_index(drop=True)


# ----------------------------------------------------------- ENSO regime

def load_nino_monthly() -> pd.Series:
    """Niño3.4 anomaly รายเดือน (index = วันที่ 1 ของเดือน)."""
    s = pd.read_csv(NINO_FILE, parse_dates=["date"], index_col="date")["nino34_anom"]
    return s.sort_index()


def enso_value_for_dates(dates, use: str = "lag1m") -> np.ndarray:
    """ค่า Niño3.4 anomaly ต่อ "วันออกพยากรณ์".

    use="lag1m"      : ค่า "เดือนก่อนหน้า" = ค่าที่โมเดลเห็นจริง (nino34_lag1m)
                       เลียนแบบ build_dataset.nino_lagged_daily (build_dataset.py:170)
    use="concurrent" : ค่าเดือนเดียวกับวันออกพยากรณ์ (ใช้เป็น robustness check)
    """
    dates = pd.DatetimeIndex(dates)
    nino = load_nino_monthly()
    if use == "lag1m":
        nino = nino.shift(1)  # index รายเดือนต่อเนื่อง -> เลื่อน 1 แถว = เดือนก่อนหน้า
    elif use != "concurrent":
        raise ValueError(f"use ต้องเป็น 'lag1m' หรือ 'concurrent' ได้ {use!r}")
    month_key = dates.to_period("M").to_timestamp()
    return nino.reindex(month_key).to_numpy()


def classify_enso(values, thresh: float = ENSO_THRESH) -> np.ndarray:
    """แปลง anomaly -> ป้าย regime (elnino / lanina / neutral ; nan -> unknown)."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, "neutral", dtype=object)
    out[v >= thresh] = "elnino"
    out[v <= -thresh] = "lanina"
    out[np.isnan(v)] = "unknown"
    return out


def attach_regime(pred: pd.DataFrame, use: str = "lag1m",
                  thresh: float = ENSO_THRESH) -> pd.DataFrame:
    """เพิ่มคอลัมน์ enso_anom + regime ตามวันออกพยากรณ์ของแต่ละแถว."""
    out = pred.copy()
    vals = enso_value_for_dates(out["date"], use=use)
    out["enso_anom"] = vals
    out["regime"] = classify_enso(vals, thresh)
    return out


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # 1) classify_enso (ฟังก์ชันบริสุทธิ์)
    lab = classify_enso([0.6, -0.7, 0.1, np.nan])
    assert list(lab) == ["elnino", "lanina", "neutral", "unknown"], list(lab)
    print("[OK] classify_enso: 0.6->elnino, -0.7->lanina, 0.1->neutral, nan->unknown")

    # 2) ENSO lag ใช้ไฟล์จริง: ก.พ.1950 lag1m -> ค่า ม.ค.1950 (-1.99) ; concurrent -> ก.พ. (-1.69)
    if NINO_FILE.exists():
        v_lag = enso_value_for_dates(pd.to_datetime(["1950-02-15"]), use="lag1m")[0]
        v_con = enso_value_for_dates(pd.to_datetime(["1950-02-15"]), use="concurrent")[0]
        assert abs(v_lag - (-1.99)) < 1e-9, v_lag
        assert abs(v_con - (-1.69)) < 1e-9, v_con
        print(f"[OK] ENSO lag1m(ก.พ.1950)={v_lag} (=ม.ค.) | concurrent={v_con} (=ก.พ.)")
    else:
        print(f"[skip] ไม่พบ {NINO_FILE}")

    # 3) predictions header ตรงสเปก
    if PRED_FILE.exists():
        head = pd.read_csv(PRED_FILE, nrows=5)
        assert list(head.columns) == PRED_COLUMNS, list(head.columns)
        print("[OK] predictions.csv header ถูกต้อง:", PRED_COLUMNS)
    else:
        print(f"[skip] ไม่พบ {PRED_FILE}")

    print("[OK] io_utils self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
