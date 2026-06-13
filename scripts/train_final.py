"""
เทรนโมเดล "ตัวจบ" สำหรับใช้งานจริง (production) แล้ว serialize เป็นไฟล์ .pkl

ต่างจาก train.py:
  train.py        = harness วัดผล (rolling-origin CV) — เทรนใหม่ทุก fold เพื่อ "ประเมิน"
                    ไม่ได้เก็บโมเดลไว้ใช้
  train_final.py  = เทรน 1 ครั้งบน "ข้อมูลทั้งหมดที่มี" ต่อ lead แล้วเก็บโมเดล
                    (estimator + Platt calibrator) ลงดิสก์ ให้ predict.py โหลดไปทำนาย

ใช้ขั้นตอนเดียวกับที่ train.py วัดผล (fit_calibrated_model จาก train.py)
-> โมเดลที่ deploy = ขั้นตอนที่ตัวเลข BSS อธิบาย ไม่ drift.
โมเดล production = logistic_balanced_cal (จากบันทึกการทดลอง: แกร่งสุด ชนะ persistence
ที่ lead 2-4) ; ตามดีไซน์ มันกัน block ท้าย ~20% ไว้ recalibrate เหมือนตอนวัดผล.

ใช้งาน:  python train_final.py        # เทรน + เซฟ models/heatwave_y_rm_lead{L}.pkl
         python train_final.py test   # self-test กับข้อมูลสังเคราะห์ (ไม่แตะข้อมูลจริง)
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import FEATURES, LEADS, PRIMARY_TARGET, fit_calibrated_model

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "dataset.csv"
MODEL_DIR = ROOT / "models"

PROD_MODEL = "logistic_balanced"          # serve เป็น logistic_balanced_cal (หลัง Platt)
ARTIFACT_VERSION = 1


def artifact_path(lead: int) -> Path:
    return MODEL_DIR / f"heatwave_{PRIMARY_TARGET}_lead{lead}.pkl"


def fit_one(df: pd.DataFrame, lead: int) -> dict:
    """เทรนโมเดล production ของ lead หนึ่ง -> dict artifact พร้อม serialize."""
    col = f"{PRIMARY_TARGET}_l{lead}"
    sub = df[FEATURES + [col]].dropna().sort_index()
    X = sub[FEATURES].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)

    fitted = fit_calibrated_model(PROD_MODEL, X, y)
    if fitted is None:
        raise RuntimeError(
            f"lead {lead}: เทรนไม่ได้ (ข้อมูล {len(sub)} แถวสั้นไป หรือ calib block คลาสเดียว)"
        )
    estimator, calibrator = fitted

    # ช่วง day-of-year ของ "วันออกพยากรณ์" ที่โมเดลเคยเทรน — predict.py ใช้กันการทำนาย
    # นอกฤดูที่โมเดลไม่เคยเห็น (ข้อมูลรอบนี้มีแค่ ม.ค.-ก.ค. -> issue date ปลายฤดูถูกตัด)
    issue_doy = sub.index.dayofyear

    return {
        "artifact_version": ARTIFACT_VERSION,
        "estimator": estimator,         # sklearn Pipeline (StandardScaler + LogisticRegression)
        "calibrator": calibrator,       # PlattCalibrator ที่ fit แล้ว
        "features": list(FEATURES),     # ลำดับคอลัมน์ที่ป้อนโมเดล (ต้องตรงตอน predict)
        "model_name": f"{PROD_MODEL}_cal",
        "target": PRIMARY_TARGET,
        "target_desc": "regional-mean Tmax > p90 ราย doy, ติดต่อกัน >=3 วัน ภายในสัปดาห์เป้าหมาย",
        "lead_weeks": lead,
        "train_start": str(sub.index.min().date()),
        "train_end": str(sub.index.max().date()),
        "n_train": int(len(sub)),
        "base_rate": float(y.mean()),   # ฐานภูมิอากาศ — predict ใช้ตีความ "สูง/ต่ำกว่าปกติ"
        "train_issue_doy_min": int(issue_doy.min()),
        "train_issue_doy_max": int(issue_doy.max()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def train_final(verbose: bool = True) -> list[Path]:
    df = pd.read_csv(DATASET, parse_dates=["date"], index_col="date")
    MODEL_DIR.mkdir(exist_ok=True)
    if verbose:
        print(f"=== train_final: {PROD_MODEL}_cal บน {len(df)} วัน | leads={LEADS} ===")

    written = []
    for lead in LEADS:
        art = fit_one(df, lead)
        path = artifact_path(lead)
        with open(path, "wb") as f:
            pickle.dump(art, f)
        written.append(path)
        if verbose:
            print(f"  lead {lead}: n={art['n_train']}, base_rate={art['base_rate']:.3f}, "
                  f"train {art['train_start']}..{art['train_end']} -> {path.name}")
    if verbose:
        print(f"[OK] เซฟ {len(written)} โมเดลที่ {MODEL_DIR}")
    return written


def _selftest() -> None:
    """พิสูจน์ว่า artifact ทำนายได้ + ขั้นตอนตรงกับ fit_calibrated_model (ไม่ drift)."""
    rng = np.random.default_rng(3)
    dates = []
    for yr in range(2000, 2016):
        dates.extend(pd.date_range(f"{yr}-01-01", f"{yr}-07-31", freq="D"))
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    doy = idx.dayofyear.to_numpy()
    sm1 = 0.35 + 0.05 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 0.02, n)
    season = np.exp(-((doy - 105) ** 2) / (2 * 30 ** 2))
    p_true = 1 / (1 + np.exp(-(-4.0 + 3.0 * season - 60.0 * (sm1 - 0.35))))
    y_event = (rng.random(n) < p_true).astype(float)

    df = pd.DataFrame(index=idx)
    for c in FEATURES:
        df[c] = rng.normal(0, 1, n)
    df["sm1"] = sm1
    df[f"{PRIMARY_TARGET}_l2"] = y_event
    df.index.name = "date"

    art = fit_one(df, 2)
    # 1) artifact ทำนายได้และอยู่ใน [0,1]
    X = df[FEATURES].to_numpy(dtype=float)[:50]
    p = art["calibrator"].transform(art["estimator"].predict_proba(X)[:, 1])
    assert p.shape == (50,) and p.min() >= 0 and p.max() <= 1, "ผลทำนายต้องเป็นความน่าจะเป็น"

    # 2) ขั้นตอน predict ของ artifact = fit_calibrated_model เป๊ะ (ไม่ drift จาก train.py)
    sub = df[FEATURES + [f"{PRIMARY_TARGET}_l2"]].dropna().sort_index()
    est2, cal2 = fit_calibrated_model(
        PROD_MODEL, sub[FEATURES].to_numpy(float), sub[f"{PRIMARY_TARGET}_l2"].to_numpy(float)
    )
    p2 = cal2.transform(est2.predict_proba(X)[:, 1])
    assert np.allclose(p, p2), "artifact ต้องให้ผลเท่ากับ fit_calibrated_model (กัน drift)"

    # 3) metadata ครบ
    for k in ("features", "model_name", "lead_weeks", "base_rate", "train_end"):
        assert k in art, f"artifact ขาด metadata: {k}"
    assert art["features"] == list(FEATURES)
    print(f"[OK] artifact ทำนายได้ + ตรงกับ fit_calibrated_model | "
          f"model={art['model_name']}, base_rate={art['base_rate']:.3f}")
    print("[OK] self-test ผ่านทั้งหมด")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
    else:
        train_final()
