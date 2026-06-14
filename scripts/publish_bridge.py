"""Bridge: predict -> validate (gate) -> distribute ไป dev/prod sink."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import validate_contract as vc  # noqa: E402

DOCS_JSON = ROOT / "docs" / "forecast_provinces.json"


def default_frontend() -> Path:
    base = Path(os.environ["BRIDGE_FRONTEND_DIR"]) if os.environ.get("BRIDGE_FRONTEND_DIR") \
        else ROOT.parent / "HeatMAP_Frontend" / "public"
    return base / "forecast_provinces.json"


def default_contract() -> Path:
    base = Path(os.environ["BRIDGE_CONTRACT_DIR"]) if os.environ.get("BRIDGE_CONTRACT_DIR") \
        else ROOT.parent / "heatwave-contract"
    return base / "forecast_provinces.json"


def validate_file(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    warn = vc.check_staleness(obj)
    if warn:
        print(warn)
    errs = vc.validate_contract(obj)
    if errs:
        print(f"[FAIL] validate ไม่ผ่าน {len(errs)} ข้อ — ยกเลิก distribute:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print(f"[OK] validate ผ่าน: {len(obj['provinces'])} จังหวัด")
    return obj


def sync(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[OK] sync -> {dst}")


def publish_contract(contract_json: Path) -> None:
    repo = contract_json.parent
    if not (repo / ".git").exists():
        print(f"[FAIL] ไม่พบ git repo ที่ {repo} — สร้าง/clone contract repo ก่อน")
        raise SystemExit(1)
    sync(DOCS_JSON, contract_json)
    subprocess.run(["git", "-C", str(repo), "add", "forecast_provinces.json"], check=True)
    unchanged = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 0
    if unchanged:
        print("[ข้าม] contract ไม่เปลี่ยน — ไม่ commit/push")
        return
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "data: update forecast_provinces.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "push"], check=True)
    print("[OK] push contract -> Pages")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Pipeline bridge: predict -> validate -> distribute")
    ap.add_argument("--no-predict", action="store_true", help="ข้าม predict, ใช้ docs/forecast_provinces.json เดิม")
    ap.add_argument("--publish", action="store_true", help="sync เข้า contract repo แล้ว git push (prod)")
    ap.add_argument("--frontend", type=Path, default=None, help="path ปลายทาง dev sink (override)")
    args = ap.parse_args()

    if not args.no_predict:
        sys.path.insert(0, str(ROOT / "scripts"))
        import predict_provinces
        predict_provinces.predict(verbose=True)
    elif not DOCS_JSON.exists():
        print(f"[FAIL] ไม่มี {DOCS_JSON} แต่ใช้ --no-predict")
        return 1

    validate_file(DOCS_JSON)
    sync(DOCS_JSON, args.frontend or default_frontend())

    if args.publish:
        publish_contract(default_contract())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
