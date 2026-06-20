"""Stress / Out-of-Distribution (OOD) test สำหรับโมเดลพยากรณ์คลื่นความร้อน (lead 2)

ออกแบบตามโจทย์: โมเดลเทรนบนอุณหภูมิช่วง ~22-35 องศาC (ดู feature_ranges.csv)
แล้วลองป้อนค่าอุณหภูมิสุดขั้ว 45 / 48 / 50 องศาC + ดันสัญญาณความร้อนอื่นให้สุด
เพื่อตอบคำถามว่า "โมเดลยังตอบสมเหตุสมผลไหมเมื่อเจอข้อมูลนอกช่วงที่เคยเห็น"

เกณฑ์ผ่าน (sanity):
  1) ความน่าจะเป็นอยู่ในช่วง [0,1] เสมอ (ไม่ระเบิด / ไม่ NaN)
  2) ยิ่งร้อนขึ้น ความน่าจะเป็นต้องไม่ลดลง (monotonic non-decreasing) — ทิศทางถูกต้อง
  3) ที่อุณหภูมิสุดขั้ว ความน่าจะเป็นต้องสูงกว่า base rate อย่างชัดเจน

หมายเหตุ: นี่คือการทดสอบ "พฤติกรรมโมเดลต่อ input นอกช่วง" ไม่ใช่ input-gate
(plausibility gate ใน units_utils จะ "ปฏิเสธ" ค่าผิดปกติก่อนถึงโมเดล — คนละเรื่องกัน)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "models" / "heatwave_prov_lead2.pkl"
RANGES = ROOT / "outputs" / "analysis" / "data_eval_fullyear" / "feature_ranges.csv"


def _baseline_vector(features: list[str], ranges: pd.DataFrame) -> dict:
    """สร้างเวกเตอร์ baseline จากค่า median ของแต่ละฟีเจอร์ (เป็น 'วันปกติ')."""
    med = ranges.set_index("feature")["median"].to_dict()
    vec = {f: float(med.get(f, 0.0)) for f in features}
    # ตั้ง one-hot ภูมิภาคให้เป็นภาคเหนือ (ตัวอย่างจุดร้อน) ; ที่เหลือ 0
    for f in features:
        if f.startswith("region_"):
            vec[f] = 0.0
    if "region_North" in vec:
        vec["region_North"] = 1.0
    # lat/lon ตัวแทน (ภาคเหนือ ~ 18.8N, 99E)
    if "lat" in vec:
        vec["lat"] = 18.8
    if "lon" in vec:
        vec["lon"] = 99.0
    return vec


def _predict(art, X: np.ndarray) -> float:
    p_raw = art["estimator"].predict_proba(X)[:, 1]
    return float(art["calibrator"].transform(p_raw)[0])


def main() -> None:
    print("=" * 64)
    print("[stress] OOD test — lead 2 (logistic_balanced_cal)")
    print("=" * 64)

    art = pickle.load(open(ART, "rb"))
    features = art["features"]
    base_rate = float(art["base_rate"])
    ranges = pd.read_csv(RANGES)
    base = _baseline_vector(features, ranges)

    train_tmax_max = float(ranges.set_index("feature").loc["tmax_rm", "max"])
    print(f"  ฟีเจอร์: {len(features)} ตัว | base_rate(train) = {base_rate:.3f}")
    print(f"  tmax_rm สูงสุดที่เคยเห็นตอนเทรน = {train_tmax_max:.1f}°C")
    print()

    # ไล่อุณหภูมิจากในช่วง -> นอกช่วง (สุดขั้ว)
    sweep = [25, 30, 35, 40, 45, 48, 50]
    probs = []
    print(f"  {'tmax(°C)':>9} | {'in-range?':>9} | {'P(heatwave)':>11} | vs base")
    print("  " + "-" * 50)
    for t in sweep:
        v = dict(base)
        # ดันสัญญาณความร้อนทั้งหมดให้สอดคล้องกับ 'ร้อนจัด'
        v["tmax_rm"] = float(t)
        v["tmax_mean7"] = float(t)
        if t >= 35:  # ร้อนจัด -> สถานะ heatwave วันนี้ + สัดส่วนวันร้อนเต็ม
            v["in_hw_today"] = 1.0
            v["hot_frac7"] = 1.0
        X = np.array([[v[f] for f in features]], dtype=float)
        p = _predict(art, X)
        probs.append(p)
        in_range = "yes" if t <= train_tmax_max else "OOD"
        ratio = p / base_rate if base_rate else float("nan")
        print(f"  {t:>9} | {in_range:>9} | {p:>11.4f} | x{ratio:>4.1f}")

    probs = np.array(probs)

    # ----- เกณฑ์ผ่าน -----
    print()
    ok_range = bool(np.all((probs >= 0) & (probs <= 1)) and not np.any(np.isnan(probs)))
    # monotonic non-decreasing (อนุญาต tolerance เล็กน้อยจาก calibrator)
    diffs = np.diff(probs)
    ok_mono = bool(np.all(diffs >= -1e-6))
    ok_extreme = bool(probs[-1] > base_rate * 2)

    print(f"  [{'OK' if ok_range else 'FAIL'}] ความน่าจะเป็นอยู่ใน [0,1] ทุกจุด, ไม่มี NaN")
    print(f"  [{'OK' if ok_mono else 'FAIL'}] ยิ่งร้อนยิ่งไม่ลด (monotonic non-decreasing)")
    print(f"  [{'OK' if ok_extreme else 'FAIL'}] ที่ 50°C ความน่าจะเป็น ({probs[-1]:.3f}) > 2x base_rate ({2*base_rate:.3f})")

    assert ok_range, "ความน่าจะเป็นหลุดช่วง [0,1] หรือเป็น NaN"
    assert ok_mono, f"ไม่ monotonic: {probs.tolist()}"
    assert ok_extreme, "ที่อุณหภูมิสุดขั้ว ความน่าจะเป็นไม่สูงพอ"
    print("\n[OK] โมเดลตอบสนองต่อข้อมูลนอกช่วงอย่างสมเหตุสมผล (เสถียร + ทิศทางถูก)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
