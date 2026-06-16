"""Audit: รันทุกเช็ค (5 หมวด) บน contract ที่จะ publish -> รายงาน go/no-go (markdown)."""
from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

from checks import FRESHNESS_PLAUSIBILITY, PASS, WARN, FAIL, CheckResult
from data_quality import DATA_QUALITY
from skill import SKILL
from comms import COMMS

ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACT = ROOT / "docs" / "forecast_provinces.json"
OUT_DIR = ROOT / "docs" / "readiness"


def run_all(obj: dict) -> list[CheckResult]:
    results = []
    for fn in FRESHNESS_PLAUSIBILITY + DATA_QUALITY:
        results.append(fn(obj))          # เช็คที่รับ contract
    for fn in SKILL + COMMS:
        results.append(fn())             # เช็คที่อ่านไฟล์ระบบ (ไม่รับ contract)
    return results


def render_report(results: list[CheckResult]) -> tuple[str, bool]:
    blockers = [r for r in results if r.failed_block()]
    go = not blockers
    lines = [f"# Production Readiness Audit — {date.today().isoformat()}", ""]
    lines.append(f"**ผล: {'✅ GO (พร้อม)' if go else '🔴 NO-GO (ยังไม่พร้อม)'}**")
    if blockers:
        lines.append(f"\nblocker {len(blockers)} ข้อ (ต้องแก้ก่อน publish):")
        for r in blockers:
            lines.append(f"- 🔴 [{r.category}] {r.name}: {r.detail}")
    lines += ["", "| หมวด | เช็ค | สถานะ | รายละเอียด |", "| --- | --- | --- | --- |"]
    icon = {PASS: "✅", WARN: "⚠️", FAIL: "🔴"}
    for r in results:
        block = " (blocking)" if r.blocking else ""
        lines.append(f"| {r.category} | {r.name}{block} | {icon[r.status]} {r.status} | {r.detail} |")
    return "\n".join(lines) + "\n", go


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = Path(argv[1]) if len(argv) > 1 else CONTRACT
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = run_all(obj)
    report, go = render_report(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"AUDIT-{date.today().isoformat()}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[เขียนรายงาน] {out}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
