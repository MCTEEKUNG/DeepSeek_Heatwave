from bot import handlers

WEB = "https://heat-map-frontend.vercel.app/map"
FORECAST = {"provinces": [
    {"name_th": "เชียงใหม่", "issue_date": "2026-05-31",
     "forecasts": [{"lead_weeks": 2, "risk_level_th": "สูง", "ratio_vs_normal": 3.0}]}
]}


def test_needs_forecast_only_for_weekly():
    assert handlers.needs_forecast({"type": "postback", "postback": {"data": "action=weekly"}}) is True
    assert handlers.needs_forecast({"type": "postback", "postback": {"data": "action=about"}}) is False
    assert handlers.needs_forecast({"type": "follow"}) is False


def test_follow_returns_welcome():
    msgs = handlers.build_reply({"type": "follow"}, None, WEB)
    assert msgs and "สวัสดี" in msgs[0]["text"]


def test_about_returns_about():
    msgs = handlers.build_reply(
        {"type": "postback", "postback": {"data": "action=about"}}, None, WEB)
    assert msgs and "วิธีอ่าน" in msgs[0]["text"]


def test_weekly_with_forecast_lists_province():
    msgs = handlers.build_reply(
        {"type": "postback", "postback": {"data": "action=weekly"}}, FORECAST, WEB)
    assert msgs and "เชียงใหม่" in msgs[0]["text"]


def test_weekly_without_forecast_is_graceful():
    msgs = handlers.build_reply(
        {"type": "postback", "postback": {"data": "action=weekly"}}, None, WEB)
    assert msgs and "ยังไม่พร้อม" in msgs[0]["text"]


def test_unknown_event_returns_none():
    assert handlers.build_reply({"type": "message", "message": {"type": "image"}}, None, WEB) is None
