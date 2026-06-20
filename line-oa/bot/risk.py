from __future__ import annotations

LEVEL_ORDER = {"ต่ำ": 0, "ปกติ": 1, "ค่อนข้างสูง": 2, "สูง": 3}


def province_max_risk(province: dict, leads=(2, 3, 4)):
    """คืน (level_th, max_ratio, lead_at_max) จาก lead ที่ระดับสูงสุดในช่วง leads.
    tie-break ด้วย ratio ; คืน None ถ้าไม่มี forecast ในช่วง."""
    best = None  # (ordinal, ratio, level_th, lead)
    for fc in province.get("forecasts", []):
        if fc.get("lead_weeks") not in leads:
            continue
        level = fc.get("risk_level_th", "")
        ordinal = LEVEL_ORDER.get(level, 0)
        ratio = float(fc.get("ratio_vs_normal", 0.0))
        cand = (ordinal, ratio, level, int(fc.get("lead_weeks")))
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if best is None:
        return None
    return (best[2], best[1], best[3])


def select_risky_provinces(forecast: dict, leads=(2, 3, 4), min_ordinal: int = 2) -> dict:
    high, elevated = [], []
    for prov in forecast.get("provinces", []):
        res = province_max_risk(prov, leads)
        if res is None:
            continue
        level_th, ratio, lead = res
        if LEVEL_ORDER.get(level_th, 0) < min_ordinal:
            continue
        row = {"name_th": prov.get("name_th", "?"), "ratio": ratio, "lead": lead}
        (high if level_th == "สูง" else elevated).append(row)
    high.sort(key=lambda r: r["ratio"], reverse=True)
    elevated.sort(key=lambda r: r["ratio"], reverse=True)
    return {
        "issue_date": _issue_date(forecast),
        "high": high,
        "elevated": elevated,
        "warnings": forecast.get("warnings", []),
    }


def _issue_date(forecast: dict) -> str:
    provs = forecast.get("provinces", [])
    return provs[0].get("issue_date", "") if provs else ""
