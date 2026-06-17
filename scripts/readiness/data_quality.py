"""เช็คคุณภาพ contract ที่จะ publish: prob/base_rate ในช่วง, ไม่มี NaN/None, leads ครบ."""
from __future__ import annotations
import math
import sys

from checks import CheckResult, PASS, WARN, FAIL  # รันแบบ script จาก scripts/readiness/

EXPECTED_LEADS = {2, 3, 4, 5, 6}


def _num_ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def check_no_nan_probs(obj: dict) -> CheckResult:
    cat = "data_quality"
    bad = []
    for p in obj.get("provinces") or []:
        for f in p.get("forecasts") or []:
            prob = f.get("probability")
            if not (_num_ok(prob) and 0 <= prob <= 1):
                bad.append(f"{p.get('code')} lead{f.get('lead_weeks')}={prob!r}")
    if bad:
        return CheckResult("no_nan_probs", cat, FAIL,
                           f"probability เสีย/นอกช่วง {len(bad)} จุด: {bad[:5]}", blocking=True)
    return CheckResult("no_nan_probs", cat, PASS, "probability ครบและอยู่ใน [0,1]", blocking=True)


def check_leads_complete(obj: dict) -> CheckResult:
    cat = "data_quality"
    bad = []
    for p in obj.get("provinces") or []:
        leads = {f.get("lead_weeks") for f in (p.get("forecasts") or [])}
        if leads != EXPECTED_LEADS:
            bad.append(f"{p.get('code')}={sorted(x for x in leads if x is not None)}")
    if bad:
        return CheckResult("leads_complete", cat, FAIL,
                           f"leads ไม่ครบ {sorted(EXPECTED_LEADS)} ที่ {len(bad)} จังหวัด: {bad[:5]}", blocking=True)
    return CheckResult("leads_complete", cat, PASS, f"ทุกจังหวัดมี leads ครบ {sorted(EXPECTED_LEADS)}", blocking=True)


def check_mjo_warning(obj: dict) -> CheckResult:
    """ถ้ามี warning MJO impute -> WARN (ไม่บล็อก แต่ต้องรู้)."""
    cat = "data_quality"
    n = sum(1 for p in (obj.get("provinces") or [])
            if any("MJO" in w for w in (p.get("warnings") or [])))
    if n:
        return CheckResult("mjo_warning", cat, WARN, f"{n} จังหวัดใช้ค่า MJO กลาง (impute) — แหล่ง MJO อาจล่าช้า")
    return CheckResult("mjo_warning", cat, PASS, "ไม่มี MJO impute")


DATA_QUALITY = [check_no_nan_probs, check_leads_complete, check_mjo_warning]


def _selftest() -> None:
    good = {"provinces": [{"code": "BKK", "warnings": [],
            "forecasts": [{"lead_weeks": L, "probability": 0.3} for L in (2, 3, 4, 5, 6)]}]}
    assert check_no_nan_probs(good).status == PASS
    assert check_leads_complete(good).status == PASS
    assert check_mjo_warning(good).status == PASS
    bad = {"provinces": [{"code": "BKK", "warnings": ["MJO ไม่อัปเดต ..."],
           "forecasts": [{"lead_weeks": 2, "probability": float("nan")}]}]}
    assert check_no_nan_probs(bad).status == FAIL
    assert check_leads_complete(bad).status == FAIL
    assert check_mjo_warning(bad).status == WARN
    print("[OK] data_quality.py self-test ผ่าน")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _selftest()
