"""Operational verification updater — wire the live track-record loop.

Run WEEKLY, after publish_bridge.py has archived this week's forecast:

  python scripts/verify/update_verification.py            # local: writes docs/verification.json
  python scripts/verify/update_verification.py --publish  # CI: also push to heatwave-contract

Pipeline:
  1. verify_closed_windows() — score every archived forecast whose target
     window has now closed (using recent observed ERA5), appending real
     (prediction vs. actual) pairs to operational_pairs.csv. Idempotent.
  2. If operational_pairs.csv has any rows: score_operational.score() →
     scorecard/reliability, then export_verification_json.export() →
     docs/verification.json, then (optionally) push to the contract repo.
  3. If no operational pairs exist yet: do nothing — the frontend keeps
     showing "Track record is building" honestly.

This is OPERATIONAL data only (real weekly runs vs. what happened), never the
retrospective backtest. See export_verification_json.py for why that matters.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify.archive import verify_closed_windows
from verify.score_operational import score
from verify.export_verification_json import export

OBSERVED_DIR = ROOT / "data" / "raw_recent"
VERIFY_DIR = ROOT / "outputs" / "operational" / "verification"
OP_PAIRS = VERIFY_DIR / "operational_pairs.csv"
OP_SCORE_DIR = VERIFY_DIR / "operational"   # operational scorecard/reliability (แยกจาก backtest)
DOCS_VERIFY_JSON = ROOT / "docs" / "verification.json"


def _publish_to_contract() -> None:
    """sync docs/verification.json -> contract repo แล้ว git push (GitHub Pages)."""
    base = os.environ.get("BRIDGE_CONTRACT_DIR")
    repo = Path(base) if base else ROOT.parent / "heatwave-contract"
    if not (repo / ".git").exists():
        print(f"[ข้าม] ไม่พบ contract repo ที่ {repo} — ไม่ push verification.json")
        return
    shutil.copyfile(DOCS_VERIFY_JSON, repo / "verification.json")
    subprocess.run(["git", "-C", str(repo), "add", "verification.json"], check=True)
    staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 1
    if not staged:
        print("[ข้าม] verification.json ไม่เปลี่ยน — ไม่ push")
        return
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "data: update verification.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "push"], check=True)
    print("[OK] push verification.json -> Pages")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="อัปเดต operational track record + verification.json")
    ap.add_argument("--observed", type=Path, default=OBSERVED_DIR,
                    help="โฟลเดอร์ observed (มี tmax_thailand/ + soil_moisture_thailand/)")
    ap.add_argument("--publish", action="store_true",
                    help="push verification.json เข้า contract repo (prod)")
    args = ap.parse_args()

    # 1. score window ที่ปิดแล้ว -> append operational_pairs.csv (best-effort, ไม่บล็อก)
    try:
        n = verify_closed_windows(args.observed)
        print(f"[verify] เพิ่ม {n} pairs จาก window ที่ปิดใหม่")
    except FileNotFoundError as e:
        print(f"[ข้าม] ไม่พบ observed data ({e}) — ยังไม่ score")
    except Exception as e:
        print(f"[ข้าม] verify_closed_windows ล้มเหลว (ไม่บล็อก): {e}")

    # 2. ยังไม่มี operational pairs -> หน้าเว็บคง "building" อย่างซื่อสัตย์
    if not OP_PAIRS.exists():
        print("[ข้าม] ยังไม่มี operational_pairs.csv — track record ยังไม่เริ่ม (หน้าเว็บ = building)")
        return 0

    # 3. score + export (best-effort — verification ไม่ใช่ critical path ของ forecast)
    try:
        score(OP_PAIRS, out_dir=OP_SCORE_DIR)
        export(
            pairs_path=OP_PAIRS,
            scorecard_path=OP_SCORE_DIR / "scorecard.csv",
            reliability_path=OP_SCORE_DIR / "reliability.csv",
            out_path=DOCS_VERIFY_JSON,
        )
    except Exception as e:
        print(f"[ข้าม] score/export ล้มเหลว (ไม่บล็อก): {e}")
        return 0

    if args.publish:
        _publish_to_contract()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
