"""เช็คว่าโมเดลยังมี skill: อ่าน BSS ที่ analysis รันไว้ (ไม่ retrain). WARN ถ้าหาไฟล์ไม่เจอ.

contract เป็น per-province -> ใช้ outputs/analysis/provinces_pooled_bss.csv (bss ราย lead).
หมายเหตุ: outputs/ ถูก gitignore -> ในเครื่องที่รัน analysis แล้วเท่านั้นจึงจะ PASS ; ถ้าไม่พบ = WARN
(ไม่ block) เพราะ skill เป็นคุณสมบัติของโมเดล ไม่ใช่ของ contract แต่ละไฟล์.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

from checks import CheckResult, PASS, WARN, FAIL

ROOT = Path(__file__).resolve().parent.parent.parent
BSS_CSV = ROOT / "outputs" / "analysis" / "provinces_pooled_bss.csv"


def check_bss_positive() -> CheckResult:
    cat = "skill"
    if not BSS_CSV.exists():
        return CheckResult("bss_positive", cat, WARN,
                           f"ไม่พบ {BSS_CSV.name} (outputs/ gitignored) — รัน analysis เพื่อยืนยัน skill ก่อน publish")
    rows = []
    with open(BSS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(float(row["lead"])), float(row["bss"])))
            except (ValueError, TypeError, KeyError):
                pass
    if not rows:
        return CheckResult("bss_positive", cat, WARN, f"อ่านค่า bss จาก {BSS_CSV.name} ไม่ได้")
    neg = [(lead, b) for lead, b in rows if b <= 0]
    if neg:
        leads = ", ".join(f"lead{l}={b:.3f}" for l, b in neg)
        return CheckResult("bss_positive", cat, WARN,
                           f"พบ BSS <= 0: {leads} — skill อ่อน/ติดลบ (โมเดลอาจไม่ชนะ climatology)")
    worst = min(b for _, b in rows)
    return CheckResult("bss_positive", cat, PASS,
                       f"BSS เป็นบวกทุก lead ({len(rows)} lead, ต่ำสุด {worst:.3f}) — โมเดลชนะ climatology")


SKILL = [check_bss_positive]


def _selftest() -> None:
    r = check_bss_positive()
    assert r.status in (PASS, WARN, FAIL)
    print(f"[OK] skill.py self-test ผ่าน (status={r.status}: {r.detail})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
