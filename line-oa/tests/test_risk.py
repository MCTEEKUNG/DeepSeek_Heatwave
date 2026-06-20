import json
from pathlib import Path

from bot import risk

FIXTURE = Path(__file__).parent / "fixtures" / "forecast_sample.json"


def _load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_province_max_risk_picks_highest_level_in_lead_2_4():
    prov = {
        "name_th": "ทดสอบ",
        "forecasts": [
            {"lead_weeks": 2, "risk_level_th": "ค่อนข้างสูง", "ratio_vs_normal": 1.8},
            {"lead_weeks": 3, "risk_level_th": "สูง", "ratio_vs_normal": 2.7},
            {"lead_weeks": 4, "risk_level_th": "ค่อนข้างสูง", "ratio_vs_normal": 2.0},
            {"lead_weeks": 5, "risk_level_th": "สูง", "ratio_vs_normal": 9.9},
        ],
    }
    level, ratio, lead = risk.province_max_risk(prov)
    assert level == "สูง"
    assert ratio == 2.7
    assert lead == 3  # lead 5 ถูกตัด แม้ ratio สูงกว่า


def test_province_max_risk_none_when_no_leads_in_range():
    prov = {"forecasts": [{"lead_weeks": 6, "risk_level_th": "สูง", "ratio_vs_normal": 5}]}
    assert risk.province_max_risk(prov) is None


def test_select_filters_below_elevated_and_sorts():
    forecast = {
        "provinces": [
            {"name_th": "A", "issue_date": "2026-05-31", "forecasts": [
                {"lead_weeks": 2, "risk_level_th": "สูง", "ratio_vs_normal": 3.0}]},
            {"name_th": "B", "issue_date": "2026-05-31", "forecasts": [
                {"lead_weeks": 2, "risk_level_th": "ค่อนข้างสูง", "ratio_vs_normal": 2.0}]},
            {"name_th": "C", "issue_date": "2026-05-31", "forecasts": [
                {"lead_weeks": 2, "risk_level_th": "ปกติ", "ratio_vs_normal": 1.1}]},
            {"name_th": "D", "issue_date": "2026-05-31", "forecasts": [
                {"lead_weeks": 2, "risk_level_th": "สูง", "ratio_vs_normal": 4.0}]},
        ]
    }
    sel = risk.select_risky_provinces(forecast)
    assert [r["name_th"] for r in sel["high"]] == ["D", "A"]   # เรียง ratio มาก->น้อย
    assert [r["name_th"] for r in sel["elevated"]] == ["B"]
    assert "C" not in [r["name_th"] for r in sel["high"] + sel["elevated"]]
    assert sel["issue_date"] == "2026-05-31"


def test_select_on_real_fixture_runs():
    sel = risk.select_risky_provinces(_load())
    assert isinstance(sel["high"], list)
    assert isinstance(sel["elevated"], list)
    assert "issue_date" in sel
