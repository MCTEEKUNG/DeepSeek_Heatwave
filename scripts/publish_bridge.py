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
DOCS_VERIFY_JSON = ROOT / "docs" / "verification.json"


def _dest_issue_date(path: Path) -> str | None:
    """issue_date ที่ปลายทางมีอยู่ (จังหวัดแรก) — None ถ้าไม่มีไฟล์/อ่านไม่ได้."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj["provinces"][0]["issue_date"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
        return None


def default_frontend() -> Path:
    base = Path(os.environ["BRIDGE_FRONTEND_DIR"]) if os.environ.get("BRIDGE_FRONTEND_DIR") \
        else ROOT.parent / "HeatMAP_Frontend" / "public"
    return base / "forecast_provinces.json"


def default_contract() -> Path:
    base = Path(os.environ["BRIDGE_CONTRACT_DIR"]) if os.environ.get("BRIDGE_CONTRACT_DIR") \
        else ROOT.parent / "heatwave-contract"
    return base / "forecast_provinces.json"


def validate_file(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[FAIL] ไม่พบไฟล์: {path}")
        raise SystemExit(1)
    except json.JSONDecodeError as e:
        print(f"[FAIL] JSON เสีย: {path} ({e})")
        raise SystemExit(1)
    warn = vc.check_staleness(obj)
    if warn:
        print(warn)
    errs = vc.validate_contract(obj)
    if errs:
        print(f"[FAIL] validate ไม่ผ่าน {len(errs)} ข้อ — ยกเลิก distribute:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    # readiness gate (freshness/plausibility/data-quality blocking) — ชั้นที่สอง ก่อน distribute
    sys.path.insert(0, str(Path(__file__).resolve().parent / "readiness"))
    from gate import run_gate
    ok, blockers = run_gate(obj)
    if not ok:
        print(f"[FAIL] readiness gate ไม่ผ่าน {len(blockers)} ข้อ — ยกเลิก distribute:")
        for b in blockers:
            print(f"  - [{b.category}] {b.name}: {b.detail}")
        raise SystemExit(1)
    print(f"[OK] validate ผ่าน: {len(obj['provinces'])} จังหวัด + readiness gate")
    return obj


def sync(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[OK] sync -> {dst}")


def _has_unpushed(repo: Path) -> bool:
    """มี commit บน HEAD ที่ยังไม่ได้ push ขึ้น upstream ไหม (ถ้าไม่มี upstream ตั้งไว้ = ต้อง push)."""
    up = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "@{u}"],
                        capture_output=True, text=True)
    if up.returncode != 0:  # ยังไม่มี upstream — first push
        head = subprocess.run(["git", "-C", str(repo), "rev-list", "-n", "1", "HEAD"],
                              capture_output=True, text=True)
        return head.returncode == 0 and bool(head.stdout.strip())
    ahead = subprocess.run(["git", "-C", str(repo), "rev-list", "@{u}..HEAD", "--count"],
                           capture_output=True, text=True)
    return ahead.returncode == 0 and ahead.stdout.strip() not in ("", "0")


def publish_contract(contract_json: Path) -> None:
    repo = contract_json.parent
    if not (repo / ".git").exists():
        print(f"[FAIL] ไม่พบ git repo ที่ {repo} — สร้าง/clone contract repo ก่อน")
        raise SystemExit(1)
    sync(DOCS_JSON, contract_json)
    subprocess.run(["git", "-C", str(repo), "add", "forecast_provinces.json"], check=True)
    # sync verification.json ถ้ามี — ไม่บล็อก publish ถ้าไม่มีไฟล์
    if DOCS_VERIFY_JSON.exists():
        sync(DOCS_VERIFY_JSON, contract_json.parent / "verification.json")
        subprocess.run(["git", "-C", str(repo), "add", "verification.json"], check=True)
    # 0 = ไม่มี staged diff, 1 = มี staged diff
    staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 1
    if staged:
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "data: update forecast_provinces.json"], check=True)
    # push เมื่อมี commit ค้าง — กันกรณี push รอบก่อนล้มเหลวแล้ว retry แล้วเงียบ
    if _has_unpushed(repo):
        subprocess.run(["git", "-C", str(repo), "push"], check=True)
        print("[OK] push contract -> Pages")
    else:
        print("[ข้าม] contract ไม่เปลี่ยน + ไม่มี commit ค้าง — ไม่ push")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Pipeline bridge: predict -> validate -> distribute")
    ap.add_argument("--no-predict", action="store_true", help="ข้าม predict, ใช้ docs/forecast_provinces.json เดิม")
    ap.add_argument("--no-operational", dest="operational", action="store_false",
                    help="ใช้ข้อมูลเทรนเต็ม (issue_date ย้อนหลัง) แทน operational/raw_recent")
    ap.add_argument("--publish", action="store_true", help="sync เข้า contract repo แล้ว git push (prod)")
    ap.add_argument("--frontend", type=Path, default=None, help="path ปลายทาง dev sink (override)")
    ap.set_defaults(operational=True)
    args = ap.parse_args()

    if not args.no_predict:
        import predict_provinces  # lazy: ดึง deps หนัก (pandas ฯลฯ) เฉพาะตอนต้อง predict
        predict_provinces.predict(verbose=True, operational=args.operational)
    elif not DOCS_JSON.exists():
        print(f"[FAIL] ไม่มี {DOCS_JSON} แต่ใช้ --no-predict")
        return 1

    obj = validate_file(DOCS_JSON)
    new_issue = obj["provinces"][0]["issue_date"]
    dests = [args.frontend or default_frontend()]
    if args.publish:
        dests.append(default_contract())
    for dst in dests:
        old = _dest_issue_date(dst)
        if old is not None and old > new_issue:
            print(f"[FAIL] ปลายทาง {dst} มี issue_date ใหม่กว่า ({old} > {new_issue}) "
                  f"— ยกเลิก กัน publish ของเก่าทับของใหม่ (ใช้ --no-operational เฉพาะเมื่อจงใจ)")
            return 1
    sync(DOCS_JSON, args.frontend or default_frontend())

    if args.operational:
        from verify.archive import archive_forecast
        dest = archive_forecast(DOCS_JSON)
        if dest is not None:
            print(f"[OK] forecast archived -> {dest}")
        else:
            print(f"[ข้าม] forecast already archived for {new_issue}")

    if args.publish:
        # export verification.json ก่อน push — ไม่บล็อก publish ถ้า scorecard ยังไม่มี
        try:
            from verify.export_verification_json import export as _export_verify
            _export_verify()
        except Exception as _e:
            print(f"[ข้าม] verification.json export ล้มเหลว (ไม่บล็อก): {_e}")
        publish_contract(default_contract())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
