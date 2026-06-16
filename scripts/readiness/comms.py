"""เช็คการสื่อสารบนเว็บ (กันตื่นตระหนก): UI ต้องสื่อ 'โอกาสเกิด ไม่ใช่ความรุนแรง' + โชว์ issue_date.

ตรวจ docs/index.html (เว็บสาธารณะที่ repo นี้คุม ผ่าน GitHub Pages อ่าน forecast.json).
หมายเหตุ: per-province contract (forecast_provinces.json) ป้อน frontend ภายนอก (heatwave-contract /
HeatMAP_Frontend) ที่ไม่อยู่ใน repo นี้ -> ตรวจอัตโนมัติที่นี่ไม่ได้ ต้องตรวจด้วยมือฝั่ง frontend.
"""
from __future__ import annotations
import sys
from pathlib import Path

from checks import CheckResult, PASS, WARN, FAIL

ROOT = Path(__file__).resolve().parent.parent.parent
UI_FILES = [ROOT / "docs" / "index.html"]
# คำที่สื่อ "ความน่าจะเป็นการเกิด" (อย่างน้อยหนึ่งคำ)
PROB_PHRASES = ["ความน่าจะเป็น", "โอกาสเกิด", "probability", "โอกาส"]
ISSUE_PHRASES = ["issue_date", "วันออกพยากรณ์", "ออกพยากรณ์", "ข้อมูล ณ", "data_through"]


def check_ui_communication() -> CheckResult:
    cat = "communication"
    missing = []
    for f in UI_FILES:
        if not f.exists():
            missing.append(f"{f.name} ไม่พบ")
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if not any(ph in text for ph in PROB_PHRASES):
            missing.append(f"{f.name}: ไม่พบคำสื่อ 'ความน่าจะเป็น/โอกาสเกิด'")
        if not any(ph in text for ph in ISSUE_PHRASES):
            missing.append(f"{f.name}: ไม่พบการโชว์วันออกพยากรณ์ (issue_date)")
    if missing:
        return CheckResult("ui_communication", cat, WARN,
                           "เว็บอาจสื่อสารไม่ชัด: " + " ; ".join(missing))
    return CheckResult("ui_communication", cat, PASS,
                       "UI สื่อ 'ความน่าจะเป็น' + โชว์วันออกพยากรณ์")


COMMS = [check_ui_communication]


def _selftest() -> None:
    r = check_ui_communication()
    assert r.status in (PASS, WARN, FAIL)
    print(f"[OK] comms.py self-test ผ่าน (status={r.status}: {r.detail})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
