import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_contract as vc


def _good_province(pid=1):
    return {
        "id": pid, "code": "BKK", "name_th": "กรุงเทพมหานคร", "name_en": "Bangkok",
        "region": "Central", "lat": 13.75, "lon": 100.5, "issue_date": "2023-12-31",
        "forecasts": [
            {"lead_weeks": L, "probability": 0.3, "climatology_base_rate": 0.11,
             "ratio_vs_normal": 2.7, "risk_level_th": "สูง", "risk_level_en": "Elevated"}
            for L in (2, 3, 4, 5, 6)
        ],
    }


def _good_contract(n=77):
    return {
        "schema_version": 1, "model": "logistic_balanced_cal",
        "generated_at": "2026-06-14T10:00:00+00:00",
        "n_provinces": n, "provinces": [_good_province(i + 1) for i in range(n)],
    }


def test_good_contract_passes():
    assert vc.validate_contract(_good_contract()) == []


def test_wrong_schema_version():
    c = _good_contract(); c["schema_version"] = 2
    assert any("schema_version" in e for e in vc.validate_contract(c))


def test_n_provinces_mismatch():
    c = _good_contract(); c["n_provinces"] = 70
    assert any("n_provinces" in e for e in vc.validate_contract(c))


def test_wrong_count():
    assert any("77" in e for e in vc.validate_contract(_good_contract(n=76)))


def test_missing_lead():
    c = _good_contract(); c["provinces"][0]["forecasts"].pop()  # remove lead 6
    assert any("leads" in e for e in vc.validate_contract(c))


def test_probability_out_of_range():
    c = _good_contract(); c["provinces"][0]["forecasts"][0]["probability"] = 1.5
    assert any("probability" in e for e in vc.validate_contract(c))


def test_bad_risk_level():
    c = _good_contract(); c["provinces"][0]["forecasts"][0]["risk_level_en"] = "Extreme"
    assert any("risk_level_en" in e for e in vc.validate_contract(c))


def test_lat_out_of_range():
    c = _good_contract(); c["provinces"][0]["lat"] = 999
    assert any("lat" in e for e in vc.validate_contract(c))


def test_duplicate_id():
    c = _good_contract(); c["provinces"][1]["id"] = 1
    assert any("ซ้ำ" in e for e in vc.validate_contract(c))


def test_staleness_warns_when_old():
    c = _good_contract(); c["generated_at"] = "2020-01-01T00:00:00+00:00"
    assert vc.check_staleness(c) is not None


def test_staleness_quiet_when_fresh():
    from datetime import datetime, timezone
    c = _good_contract(); c["generated_at"] = datetime.now(timezone.utc).isoformat()
    assert vc.check_staleness(c) is None
