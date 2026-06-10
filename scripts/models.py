"""
ทะเบียนโมเดล (model registry) — หัวใจของการ "สลับโมเดลได้ง่าย"

ทุกโมเดลใช้ interface เดียวกัน (sklearn): .fit(X, y) / .predict_proba(X)
การเพิ่มโมเดลใหม่ = เพิ่ม 1 รายการใน MODEL_REGISTRY เท่านั้น

นโยบายเรื่อง class imbalance (สำคัญต่องานนี้):
  งานวัดผลด้วย Brier/BSS/reliability ซึ่งต้องการ "ความน่าจะเป็นที่ calibrate ดี"
  การถ่วงน้ำหนักคลาส (class_weight="balanced") จะดันความน่าจะเป็นของเหตุการณ์หายาก
  ให้สูงเกินจริงอย่างเป็นระบบ -> Brier/reliability แย่ลงทั้งที่ AUC อาจดูดี
  - ชื่อหลัก  logistic / lgbm           : "ไม่ถ่วงน้ำหนัก" (ค่าเริ่มต้นของงานนี้)
  - ชื่อเสริม logistic_balanced / lgbm_balanced / balanced_rf : ตัวถ่วงน้ำหนัก
    เก็บไว้เปรียบเทียบเป็น ablation ในเล่ม — ถ้าจะใช้รายงานจริงต้อง recalibrate
    บน validation block ที่แยกตามเวลาเสียก่อน (ทำใน train.py ไม่ใช่ที่นี่
    เพราะ CalibratedClassifierCV ใช้ KFold ภายในซึ่ง leak กับ time series)
"""
from __future__ import annotations

from typing import Callable
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

RANDOM_STATE = 42


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _logistic_balanced() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def _lgbm_params(**extra) -> dict:
    return dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
        **extra,
    )


def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(**_lgbm_params())


def _lgbm_balanced() -> LGBMClassifier:
    return LGBMClassifier(**_lgbm_params(class_weight="balanced"))


def _balanced_rf() -> BalancedRandomForestClassifier:
    return BalancedRandomForestClassifier(
        n_estimators=400,
        sampling_strategy="all",   # ตั้งชัดเจนเพื่อพฤติกรรม BRF คลาสสิก + กัน FutureWarning
        replacement=True,
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ทะเบียน: ชื่อ -> factory ที่คืน estimator ใหม่ (เพิ่มโมเดลใหม่ที่นี่)
MODEL_REGISTRY: dict[str, Callable[[], object]] = {
    # ไม่ถ่วงน้ำหนัก = ค่าเริ่มต้นของงานนี้ (probability ที่ calibrate ดี)
    "logistic": _logistic,
    "lgbm": _lgbm,
    # ถ่วงน้ำหนัก = เก็บไว้เป็น ablation (ระวัง: probability เฟ้อ ต้อง recalibrate ก่อนใช้จริง)
    "logistic_balanced": _logistic_balanced,
    "lgbm_balanced": _lgbm_balanced,
    "balanced_rf": _balanced_rf,
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_model(name: str, **overrides):
    """สร้าง estimator ตามชื่อ ; overrides = ปรับ hyperparameter เฉพาะ run นั้น."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"ไม่รู้จักโมเดล '{name}' ; ที่มี: {list_models()}")
    model = MODEL_REGISTRY[name]()
    if overrides:
        model.set_params(**overrides)
    return model


def _selftest() -> None:
    # ทดสอบ "interface" ของทะเบียน (ไม่ใช่ผลการทดลองจริง) ด้วยข้อมูล imbalanced สังเคราะห์
    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600, n_features=8, weights=[0.9, 0.1], random_state=0
    )
    print("โมเดลในทะเบียน:", list_models())
    for name in list_models():
        model = get_model(name)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2), (name, proba.shape)
        assert proba.min() >= 0 and proba.max() <= 1, name
        print(f"  {name:12s} -> fit/predict_proba OK (shape={proba.shape})")

    # ทดสอบประเด็น calibration: บนข้อมูล imbalanced (base rate ~0.1)
    # - โมเดลไม่ถ่วงน้ำหนัก: ค่าเฉลี่ย predict_proba ต้องใกล้ base rate (calibrate ดี)
    # - โมเดลถ่วงน้ำหนัก:   ค่าเฉลี่ยจะเฟ้อสูงกว่า base rate มาก (เหตุผลที่ต้อง recalibrate)
    base_rate = y.mean()
    p_plain = get_model("logistic").fit(X, y).predict_proba(X)[:, 1].mean()
    p_balanced = get_model("logistic_balanced").fit(X, y).predict_proba(X)[:, 1].mean()
    assert abs(p_plain - base_rate) < 0.03, (p_plain, base_rate)
    assert p_balanced > 1.5 * base_rate, (p_balanced, base_rate)  # เฟ้อเกิน 1.5 เท่าของฐาน
    print(f"  calibration check: base_rate={base_rate:.3f} | "
          f"logistic={p_plain:.3f} (ใกล้ฐาน ✅) | logistic_balanced={p_balanced:.3f} (เฟ้อ ตามคาด ⚠️)")

    # ทดสอบ override + error ของชื่อที่ไม่รู้จัก
    m = get_model("lgbm", n_estimators=10)
    assert m.get_params()["n_estimators"] == 10
    try:
        get_model("ไม่มีจริง")
        raise AssertionError("ควร error เมื่อชื่อไม่รู้จัก")
    except KeyError:
        pass
    print("[OK] ทะเบียนโมเดลทำงานครบ (fit, predict_proba, override, error handling)")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
