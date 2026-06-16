"""Gate: รันเฉพาะเช็ค blocking บน contract -> exit 1 ถ้ามี blocker (ใช้ก่อน publish)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

from checks import FRESHNESS_PLAUSIBILITY
from data_quality import DATA_QUALITY

ROOT = Path(__file__).resolve().parent.parent.parent
CONTRACT = ROOT / "docs" / "forecast_provinces.json"


def gate_results(obj: dict):
    return [fn(obj) for fn in FRESHNESS_PLAUSIBILITY + DATA_QUALITY]


def run_gate(obj: dict) -> tuple[bool, list]:
    results = gate_results(obj)
    blockers = [r for r in results if r.failed_block()]
    return (not blockers), blockers


def _selftest() -> None:
    from datetime import datetime, timezone, timedelta
    now_iso = datetime.now(timezone.utc).isoformat()
    # พยากรณ์สด: สร้างวันนี้ ใช้ข้อมูลล่าสุด ~16 วันก่อน (ERA5 ล่าช้า)
    op_issue = (datetime.now(timezone.utc).date() - timedelta(days=16)).isoformat()
    good = {"generated_at": now_iso, "provinces": [{"code": "BKK", "issue_date": op_issue, "warnings": [],
            "forecasts": [{"lead_weeks": L, "probability": 0.3, "risk_level_en": "Normal",
                           "ratio_vs_normal": 1.1} for L in (2, 3, 4, 5, 6)]}]}
    ok, blk = run_gate(good)
    assert ok and not blk, f"contract ดีต้องผ่าน gate: {[b.detail for b in blk]}"
    # negative: ข้อมูล 2023 ต้องโดนบล็อก
    stale = json.loads(json.dumps(good))
    stale["provinces"][0]["issue_date"] = "2023-12-31"
    ok2, blk2 = run_gate(stale)
    assert not ok2 and blk2, "ข้อมูล 2023 ต้องโดน gate บล็อก"
    # negative: probability เสีย ต้องโดนบล็อก
    badp = json.loads(json.dumps(good))
    badp["provinces"][0]["forecasts"][0]["probability"] = 1.5
    ok3, blk3 = run_gate(badp)
    assert not ok3 and blk3, "probability นอกช่วงต้องโดน gate บล็อก"
    print(f"[OK] gate.py self-test ผ่าน (good=GO, stale=NO-GO[{blk2[0].name}], badprob=NO-GO[{blk3[0].name}])")


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(argv) > 1 and argv[1] == "test":
        _selftest()
        return 0
    path = Path(argv[1]) if len(argv) > 1 else CONTRACT
    obj = json.loads(path.read_text(encoding="utf-8"))
    ok, blockers = run_gate(obj)
    if not ok:
        print(f"[GATE FAIL] blocker {len(blockers)} ข้อ — ห้าม publish:")
        for b in blockers:
            print(f"  - [{b.category}] {b.name}: {b.detail}")
        return 1
    print("[GATE OK] ผ่านเช็ค blocking ทั้งหมด")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
