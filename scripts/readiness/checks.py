"""ชุดเช็คความพร้อม production: โครงสร้างผลลัพธ์ + freshness + plausibility.
อ่าน contract JSON ที่จะ publish — ไม่ retrain. ใช้ทั้ง audit (ทุกเช็ค) และ gate (blocking).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MAX_ISSUE_AGE_DAYS = 10          # issue_date เก่ากว่านี้ = ข้อมูลไม่สด (อาจเป็น demo/backtest)
MAX_HIGH_FRACTION = 0.80         # จังหวัดที่ขึ้น High พร้อมกันเกินสัดส่วนนี้ = ควรดูด้วยตา (WARN เท่านั้น)
MAX_RATIO = 6.0                  # ratio_vs_normal เกินนี้ = ควรตรวจสอบ (WARN)


@dataclass
class CheckResult:
    name: str
    category: str
    status: str           # PASS | WARN | FAIL
    detail: str
    blocking: bool = False   # True = ถ้า FAIL ห้าม publish

    def failed_block(self) -> bool:
        return self.blocking and self.status == FAIL


def _today() -> date:
    return datetime.now(timezone.utc).date()


def check_issue_date_fresh(obj: dict) -> CheckResult:
    """Freshness: issue_date (ไม่ใช่ generated_at) ต้องไม่เก่าเกิน MAX_ISSUE_AGE_DAYS."""
    cat = "freshness"
    provs = obj.get("provinces") or []
    if not provs:
        return CheckResult("issue_date_fresh", cat, FAIL, "ไม่มี provinces", blocking=True)
    issues = {p.get("issue_date") for p in provs}
    try:
        ages = [(_today() - date.fromisoformat(d)).days for d in issues if d]
    except (ValueError, TypeError):
        return CheckResult("issue_date_fresh", cat, FAIL, f"issue_date parse ไม่ได้: {issues}", blocking=True)
    if not ages:
        return CheckResult("issue_date_fresh", cat, FAIL, "ไม่มี issue_date", blocking=True)
    worst = max(ages)
    if worst > MAX_ISSUE_AGE_DAYS:
        return CheckResult("issue_date_fresh", cat, FAIL,
                           f"issue_date เก่า {worst} วัน (เกิน {MAX_ISSUE_AGE_DAYS}) — อาจเป็นข้อมูล demo/backtest",
                           blocking=True)
    return CheckResult("issue_date_fresh", cat, PASS, f"issue_date เก่าสุด {worst} วัน", blocking=True)


def check_all_high_fraction(obj: dict) -> CheckResult:
    """Plausibility (WARN เท่านั้น — ไม่ blocking): สัดส่วนจังหวัด 'ทุก lead = High'.

    หมายเหตุสำคัญ (advisor): ห้าม blocking. ช่วง El Nino แรงจริง (เช่น 2023->ต้นปี 2024 ไทยร้อนทำลายสถิติ)
    'เกือบทุกจังหวัด High หลายสัปดาห์' = สัญญาณเตือนที่ "ถูกต้อง" และเป็นตอนที่ประชาชนต้องการเตือนที่สุด.
    ถ้า block ไว้ = fail closed ทับสัญญาณจริง (ตรงข้ามเป้าหมาย). กรณี demo 2023 ถูกจับโดย freshness แล้ว.
    ที่นี่แค่ WARN ให้คนดูยืนยันด้วยตา.
    """
    cat = "plausibility"
    provs = obj.get("provinces") or []
    if not provs:
        return CheckResult("all_high_fraction", cat, WARN, "ไม่มี provinces")
    n_all_high = 0
    for p in provs:
        fcs = p.get("forecasts") or []
        if fcs and all(f.get("risk_level_en") == "High" for f in fcs):
            n_all_high += 1
    frac = n_all_high / len(provs)
    status = WARN if frac > MAX_HIGH_FRACTION else PASS
    note = " — ตรวจด้วยตา (อาจถูกต้องถ้า El Nino แรง / อาจผิดปกติ)" if status == WARN else ""
    return CheckResult("all_high_fraction", cat, status,
                       f"{n_all_high}/{len(provs)} จังหวัด High ทุก lead ({frac:.0%}){note}")


def check_ratio_bounds(obj: dict) -> CheckResult:
    """Plausibility: ratio_vs_normal ทุกค่าควร <= MAX_RATIO (WARN ถ้าเกิน)."""
    cat = "plausibility"
    worst = 0.0
    for p in obj.get("provinces") or []:
        for f in p.get("forecasts") or []:
            r = f.get("ratio_vs_normal")
            if isinstance(r, (int, float)) and not isinstance(r, bool):
                worst = max(worst, float(r))
    if worst > MAX_RATIO:
        return CheckResult("ratio_bounds", cat, WARN,
                           f"ratio_vs_normal สูงสุด {worst} (เกิน {MAX_RATIO}) — ตรวจสอบความสมเหตุสมผล")
    return CheckResult("ratio_bounds", cat, PASS, f"ratio_vs_normal สูงสุด {worst}")


FRESHNESS_PLAUSIBILITY = [check_issue_date_fresh, check_all_high_fraction, check_ratio_bounds]


def _selftest() -> None:
    today = _today().isoformat()
    good = {"provinces": [
        {"issue_date": today, "forecasts": [{"risk_level_en": "Normal", "ratio_vs_normal": 1.2}]},
        {"issue_date": today, "forecasts": [{"risk_level_en": "Low", "ratio_vs_normal": 0.8}]},
    ]}
    assert check_issue_date_fresh(good).status == PASS
    assert check_all_high_fraction(good).status == PASS
    # negative: ข้อมูลเก่า 2023 ต้อง FAIL + blocking
    stale = {"provinces": [{"issue_date": "2023-12-31",
                            "forecasts": [{"risk_level_en": "High", "ratio_vs_normal": 4.0}]}]}
    r = check_issue_date_fresh(stale)
    assert r.status == FAIL and r.blocking, "ข้อมูลเก่าต้อง FAIL+blocking"
    # ทุกจังหวัด High ทุก lead = WARN (ไม่ใช่ FAIL/blocking — อาจเป็นสัญญาณจริงช่วง El Nino)
    allhigh = {"provinces": [{"issue_date": today, "forecasts": [{"risk_level_en": "High", "ratio_vs_normal": 5.0}]}
                             for _ in range(10)]}
    r2 = check_all_high_fraction(allhigh)
    assert r2.status == WARN and not r2.blocking, "all-High ต้อง WARN ไม่ block (กันทับสัญญาณจริง)"
    # ratio เกินเพดาน = WARN
    assert check_ratio_bounds(allhigh).status == PASS  # 5.0 <= 6.0
    big = {"provinces": [{"forecasts": [{"ratio_vs_normal": 9.9}]}]}
    assert check_ratio_bounds(big).status == WARN
    print("[OK] checks.py self-test ผ่าน (freshness + plausibility + negative cases)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
