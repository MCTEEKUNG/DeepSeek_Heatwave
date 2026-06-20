import json
import sys
from pathlib import Path

# ให้ import scripts.broadcast_weekly ได้ (line-oa root อยู่บน path เมื่อรัน pytest จาก line-oa)
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import broadcast_weekly as bw  # noqa: E402

FORECAST = {"warnings": [], "provinces": [
    {"name_th": "เชียงใหม่", "issue_date": "2026-05-31",
     "forecasts": [{"lead_weeks": 2, "risk_level_th": "สูง", "ratio_vs_normal": 3.0}]}
]}


def test_load_forecast_from_file(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(FORECAST), encoding="utf-8")
    data = bw.load_forecast(cfg=None, file_path=str(p))
    assert data["provinces"][0]["name_th"] == "เชียงใหม่"


def test_main_broadcasts_when_token_present(monkeypatch, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(FORECAST), encoding="utf-8")
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")

    sent = {}

    def fake_broadcast(token, messages, timeout=10):
        sent["token"], sent["text"] = token, messages[0]["text"]
        class R: status_code = 200; text = "{}"
        return R()

    monkeypatch.setattr(bw.line_api, "broadcast", fake_broadcast)
    rc = bw.main(["--file", str(p)])
    assert rc == 0
    assert "เชียงใหม่" in sent["text"]


def test_main_returns_1_without_token(monkeypatch, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(FORECAST), encoding="utf-8")
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    rc = bw.main(["--file", str(p)])
    assert rc == 1
