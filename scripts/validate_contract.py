"""ตรวจ contract forecast_provinces.json ก่อน distribute (hard gate)."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ALLOWED_RISK_EN = {"Low", "Normal", "Elevated", "High"}
EXPECTED_LEADS = {2, 3, 4, 5, 6}
EXPECTED_N = 77
PROVINCE_KEYS = ("id", "code", "name_th", "name_en", "region",
                 "lat", "lon", "issue_date", "forecasts")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_contract(obj) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["contract ต้องเป็น object/dict"]
    if obj.get("schema_version") != 1:
        errs.append(f"schema_version ต้อง == 1 (เจอ {obj.get('schema_version')!r})")
    model = obj.get("model")
    if not isinstance(model, str) or not model.strip():
        errs.append("model ต้องเป็น str ไม่ว่าง")
    ga = obj.get("generated_at")
    if not isinstance(ga, str):
        errs.append("generated_at ต้องเป็น str (ISO datetime)")
    else:
        try:
            datetime.fromisoformat(ga)
        except ValueError:
            errs.append(f"generated_at parse ไม่ได้: {ga!r}")

    provinces = obj.get("provinces")
    if not isinstance(provinces, list):
        errs.append("provinces ต้องเป็น list")
        return errs
    if obj.get("n_provinces") != len(provinces):
        errs.append(f"n_provinces ({obj.get('n_provinces')}) != len(provinces) ({len(provinces)})")
    if len(provinces) != EXPECTED_N:
        errs.append(f"จำนวนจังหวัดต้อง == {EXPECTED_N} (เจอ {len(provinces)})")

    seen_ids: set = set()
    for i, p in enumerate(provinces):
        tag = f"province[{i}]"
        if not isinstance(p, dict):
            errs.append(f"{tag} ต้องเป็น object")
            continue
        for k in PROVINCE_KEYS:
            if k not in p:
                errs.append(f"{tag} ขาด key '{k}'")
        pid = p.get("id")
        if isinstance(pid, int) and not isinstance(pid, bool):
            if pid in seen_ids:
                errs.append(f"{tag} id ซ้ำ: {pid}")
            seen_ids.add(pid)
            tag = f"province id={pid}"
        else:
            errs.append(f"{tag} id ต้องเป็น int")
        lat, lon = p.get("lat"), p.get("lon")
        if not (_is_num(lat) and -90 <= lat <= 90):
            errs.append(f"{tag} lat นอกช่วง [-90,90]: {lat!r}")
        if not (_is_num(lon) and -180 <= lon <= 180):
            errs.append(f"{tag} lon นอกช่วง [-180,180]: {lon!r}")
        try:
            date.fromisoformat(p.get("issue_date"))
        except (ValueError, TypeError):
            errs.append(f"{tag} issue_date parse ไม่ได้: {p.get('issue_date')!r}")

        fcs = p.get("forecasts")
        if not isinstance(fcs, list):
            errs.append(f"{tag} forecasts ต้องเป็น list")
            continue
        leads = []
        for f in fcs:
            if not isinstance(f, dict):
                errs.append(f"{tag} forecast ต้องเป็น object")
                continue
            L = f.get("lead_weeks")
            leads.append(L)
            prob = f.get("probability")
            if not (_is_num(prob) and 0 <= prob <= 1):
                errs.append(f"{tag} lead {L} probability นอกช่วง [0,1]: {prob!r}")
            br = f.get("climatology_base_rate")
            if not (_is_num(br) and 0 <= br <= 1):
                errs.append(f"{tag} lead {L} climatology_base_rate นอกช่วง [0,1]: {br!r}")
            ratio = f.get("ratio_vs_normal")
            if not (_is_num(ratio) and ratio >= 0):
                errs.append(f"{tag} lead {L} ratio_vs_normal ต้อง >= 0: {ratio!r}")
            if f.get("risk_level_en") not in ALLOWED_RISK_EN:
                errs.append(f"{tag} lead {L} risk_level_en ไม่ถูก: {f.get('risk_level_en')!r}")
            th = f.get("risk_level_th")
            if not isinstance(th, str) or not th.strip():
                errs.append(f"{tag} lead {L} risk_level_th ว่าง")
        if set(leads) != EXPECTED_LEADS:
            got = sorted(x for x in leads if x is not None)
            errs.append(f"{tag} leads ต้องเป็น {sorted(EXPECTED_LEADS)} (เจอ {got})")
    return errs


def check_staleness(obj, max_age_days: int = 7) -> str | None:
    try:
        ga_dt = datetime.fromisoformat(obj.get("generated_at"))
    except (ValueError, TypeError):
        return None
    if ga_dt.tzinfo is None:
        ga_dt = ga_dt.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - ga_dt).days
    return f"[เตือน] contract เก่า {age} วัน (generated_at {obj.get('generated_at')})" if age > max_age_days else None


def main(argv) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = Path(argv[1]) if len(argv) > 1 else \
        Path(__file__).resolve().parent.parent / "docs" / "forecast_provinces.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    warn = check_staleness(obj)
    if warn:
        print(warn)
    errs = validate_contract(obj)
    if errs:
        print(f"[FAIL] contract ไม่ผ่าน {len(errs)} ข้อ:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print(f"[OK] contract ผ่าน: {len(obj['provinces'])} จังหวัด, schema v{obj['schema_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
