"""ชุดเช็คความพร้อม production: โครงสร้างผลลัพธ์ + freshness + plausibility.
อ่าน contract JSON ที่จะ publish — ไม่ retrain. ใช้ทั้ง audit (ทุกเช็ค) และ gate (blocking).
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
# Freshness วัด "ช่องว่างระหว่างวันสร้างพยากรณ์กับวันของข้อมูลที่ใช้" (generated_at - issue_date)
# ไม่ใช่ (วันนี้ - issue_date) — เพราะตัวแยก demo(ใช้ข้อมูล 2023) ออกจากพยากรณ์สด(ERA5 ล่าช้า ~6 วัน)
# คือ "ข้อมูลล้าหลังตอนสร้างแค่ไหน": demo gap ~898 วัน vs operational ~16 วัน.
MAX_DATA_LAG_DAYS = 30           # generated_at - issue_date เกินนี้ = ใช้ข้อมูลเก่า/ผิด (block)
MAX_ISSUE_AGE_DAYS = 21          # วันนี้ - issue_date เกินนี้ = pipeline ค้าง ไม่เดินหน้า (block)
MAX_GENERATED_AGE_DAYS = 14      # วันนี้ - generated_at เกินนี้ = ไฟล์ไม่ถูก refresh (WARN)
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


def _parse_generated_date(obj: dict):
    """คืน date ของ generated_at (ISO datetime) หรือ None ถ้า parse ไม่ได้."""
    ga = obj.get("generated_at")
    try:
        return datetime.fromisoformat(ga).date()
    except (ValueError, TypeError):
        return None


def check_data_lag(obj: dict) -> CheckResult:
    """Freshness (blocking): ช่องว่าง generated_at - issue_date (ข้อมูลล้าหลังตอนสร้างแค่ไหน).

    นี่คือตัวแยก 'demo ใช้ข้อมูล 2023' (gap ~898 วัน) ออกจาก 'พยากรณ์สดที่ ERA5 ล่าช้า' (gap ~16 วัน).
    วัด today-issue_date ไม่ได้ เพราะพยากรณ์สดที่ถูกต้องก็ห่างวันนี้ ~6-16 วันตาม latency อยู่แล้ว.
    """
    cat = "freshness"
    provs = obj.get("provinces") or []
    if not provs:
        return CheckResult("data_lag", cat, FAIL, "ไม่มี provinces", blocking=True)
    gen = _parse_generated_date(obj)
    if gen is None:
        return CheckResult("data_lag", cat, FAIL,
                           f"generated_at parse ไม่ได้: {obj.get('generated_at')!r}", blocking=True)
    issues = {p.get("issue_date") for p in provs}
    try:
        lags = [(gen - date.fromisoformat(d)).days for d in issues if d]
    except (ValueError, TypeError):
        return CheckResult("data_lag", cat, FAIL, f"issue_date parse ไม่ได้: {issues}", blocking=True)
    if not lags:
        return CheckResult("data_lag", cat, FAIL, "ไม่มี issue_date", blocking=True)
    worst = max(lags)          # lag มากสุด = ข้อมูลเก่าสุดเทียบกับวันสร้าง
    if worst > MAX_DATA_LAG_DAYS:
        return CheckResult("data_lag", cat, FAIL,
                           f"ข้อมูลล้าหลัง {worst} วันตอนสร้างพยากรณ์ (generated_at - issue_date เกิน "
                           f"{MAX_DATA_LAG_DAYS}) — น่าจะใช้ข้อมูลเก่า/demo ไม่ใช่ข้อมูลล่าสุด", blocking=True)
    if worst < -2:
        return CheckResult("data_lag", cat, FAIL,
                           f"issue_date ล้ำหน้า generated_at {-worst} วัน — ผิดปกติ (ข้อมูลอนาคต?)", blocking=True)
    return CheckResult("data_lag", cat, PASS, f"ข้อมูลล้าหลังตอนสร้าง {worst} วัน (<= {MAX_DATA_LAG_DAYS})",
                       blocking=True)


def check_issue_date_current(obj: dict) -> CheckResult:
    """Freshness (blocking): วันนี้ - issue_date ล่าสุด — จับ 'pipeline ค้าง'.

    ต่างจาก check_data_lag (generated_at - issue_date) ที่จับ 'demo/ข้อมูลผิดยุค' (gap ใหญ่
    ตอนสร้าง). เคสนี้คือ pipeline รันสด (generated_at ใหม่เสมอ) แต่ issue_date ไม่ขยับตาม
    ปฏิทิน เพราะ input ค้าง — เช่น ERA5/CDS คืนข้อมูลสั้น/เก่าเงียบๆ, soil ไม่อัปเดต, หรือ
    ฟีเจอร์ที่ประกาศช้า. check_data_lag พลาดเคสนี้ตอนกลางเดือน (gap ยังไม่ถึง 30) — ตัวนี้ปิดช่องนั้น.

    เช็คตอน publish (issue_date เพิ่งคำนวณสด): พยากรณ์สุขภาพดี issue_date เก่าแค่ ~6-16 วัน
    (ERA5 latency); pipeline ค้างจะโตเรื่อยๆ ไม่หยุด -> threshold 21 แยกสองกรณีได้ชัด."""
    cat = "freshness"
    provs = obj.get("provinces") or []
    if not provs:
        return CheckResult("issue_date_current", cat, FAIL, "ไม่มี provinces", blocking=True)
    issues = {p.get("issue_date") for p in provs}
    try:
        ages = [(_today() - date.fromisoformat(d)).days for d in issues if d]
    except (ValueError, TypeError):
        return CheckResult("issue_date_current", cat, FAIL,
                           f"issue_date parse ไม่ได้: {issues}", blocking=True)
    if not ages:
        return CheckResult("issue_date_current", cat, FAIL, "ไม่มี issue_date", blocking=True)
    worst = max(ages)            # issue_date เก่าสุดเทียบวันนี้
    if worst > MAX_ISSUE_AGE_DAYS:
        return CheckResult("issue_date_current", cat, FAIL,
                           f"issue_date เก่า {worst} วันเทียบวันนี้ (เกิน {MAX_ISSUE_AGE_DAYS}) — "
                           f"pipeline อาจค้าง (input ไม่อัปเดต) แม้ generated_at จะสด", blocking=True)
    return CheckResult("issue_date_current", cat, PASS,
                       f"issue_date เก่า {worst} วันเทียบวันนี้ (<= {MAX_ISSUE_AGE_DAYS})", blocking=True)


def check_generated_recent(obj: dict) -> CheckResult:
    """Freshness (WARN): วันนี้ - generated_at — ไฟล์ถูก refresh ล่าสุดเมื่อไร (ไม่ block)."""
    cat = "freshness"
    gen = _parse_generated_date(obj)
    if gen is None:
        return CheckResult("generated_recent", cat, WARN, "generated_at parse ไม่ได้")
    age = (_today() - gen).days
    if age > MAX_GENERATED_AGE_DAYS:
        return CheckResult("generated_recent", cat, WARN,
                           f"พยากรณ์ถูกสร้างมาแล้ว {age} วัน (เกิน {MAX_GENERATED_AGE_DAYS}) — ควร re-run ให้สด")
    return CheckResult("generated_recent", cat, PASS, f"สร้างเมื่อ {age} วันก่อน")


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


FRESHNESS_PLAUSIBILITY = [check_data_lag, check_issue_date_current, check_generated_recent,
                          check_all_high_fraction, check_ratio_bounds]


def _selftest() -> None:
    from datetime import timedelta
    now_iso = datetime.now(timezone.utc).isoformat()
    today = _today()
    # พยากรณ์สดจริง: สร้างวันนี้ ใช้ข้อมูลล่าสุด ~16 วันก่อน (ERA5 ล่าช้า) -> ต้องผ่าน
    op_issue = (today - timedelta(days=16)).isoformat()
    good = {"generated_at": now_iso, "provinces": [
        {"issue_date": op_issue, "forecasts": [{"risk_level_en": "Normal", "ratio_vs_normal": 1.2}]},
        {"issue_date": op_issue, "forecasts": [{"risk_level_en": "Low", "ratio_vs_normal": 0.8}]},
    ]}
    assert check_data_lag(good).status == PASS, "พยากรณ์สด (lag 16 วัน) ต้องผ่าน data_lag"
    assert check_issue_date_current(good).status == PASS, "issue 16 วัน ต้องผ่าน issue_date_current"
    assert check_generated_recent(good).status == PASS
    assert check_all_high_fraction(good).status == PASS
    # negative: demo — สร้าง 2026 แต่ใช้ข้อมูล 2023 (gap ~898 วัน) ต้อง FAIL + blocking
    stale = {"generated_at": now_iso, "provinces": [{"issue_date": "2023-12-31",
                            "forecasts": [{"risk_level_en": "High", "ratio_vs_normal": 4.0}]}]}
    r = check_data_lag(stale)
    assert r.status == FAIL and r.blocking, "demo (ข้อมูล 2023) ต้อง FAIL+blocking"
    assert check_issue_date_current(stale).status == FAIL, "demo 2023 ต้อง FAIL issue_date_current ด้วย"
    # gap case (เหมือนบั๊ก Niño): สร้างวันนี้ แต่ issue_date ค้าง 25 วัน -> data_lag ยังผ่าน (25<30)
    # แต่ issue_date_current ต้อง FAIL+block (25>21). นี่คือช่องที่ guard ใหม่ปิด.
    stuck = {"generated_at": now_iso, "provinces": [{"issue_date": (today - timedelta(days=25)).isoformat(),
                            "forecasts": [{"risk_level_en": "Normal", "ratio_vs_normal": 1.1}]}]}
    assert check_data_lag(stuck).status == PASS, "gap 25 วัน (<30) data_lag ยังผ่าน — ช่องโหว่เดิม"
    rs = check_issue_date_current(stuck)
    assert rs.status == FAIL and rs.blocking, "pipeline ค้าง 25 วัน ต้อง FAIL+block ด้วย issue_date_current"
    # negative: generated_at เก่า -> generated_recent = WARN (ไม่ block)
    old_gen = {"generated_at": (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(),
               "provinces": [{"issue_date": (today - timedelta(days=46)).isoformat(),
                              "forecasts": [{"risk_level_en": "Normal", "ratio_vs_normal": 1.0}]}]}
    assert check_generated_recent(old_gen).status == WARN
    assert not check_generated_recent(old_gen).blocking
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
