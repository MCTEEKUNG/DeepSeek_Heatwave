"""
ทะเบียนโมเดล (model registry) — หัวใจของการ "สลับโมเดลได้ง่าย"

ทุกโมเดลใช้ interface เดียวกัน (sklearn): .fit(X, y) / .predict_proba(X)
การเพิ่มโมเดลใหม่ = เพิ่ม 1 รายการใน MODEL_REGISTRY เท่านั้น

ค่าเริ่มต้นถูกตั้งให้รับมือ "เหตุการณ์หายาก (class imbalance)" ของคลื่นความร้อน:
  - logistic / lgbm : class_weight="balanced"
  - balanced_rf      : สุ่ม under-sample คลาสเด่นในแต่ละต้นไม้ (Chen et al.) — งานวิจัยรองรับ
"""
from __future__ import annotations

from typing import Callable
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

RANDOM_STATE = 42


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def _lgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


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
    "logistic": _logistic,
    "lgbm": _lgbm,
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
