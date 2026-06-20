from bot import messages

WEB = "https://heat-map-frontend.vercel.app/map"


def _sel(high=None, elevated=None, warnings=None):
    return {"issue_date": "2026-05-31", "high": high or [], "elevated": elevated or [],
            "warnings": warnings or []}


def test_weekly_summary_lists_provinces_and_link():
    sel = _sel(high=[{"name_th": "เชียงใหม่", "ratio": 3.0, "lead": 2}],
               elevated=[{"name_th": "น่าน", "ratio": 2.0, "lead": 3}])
    msg = messages.build_weekly_summary(sel, WEB)
    assert "เชียงใหม่" in msg
    assert "น่าน" in msg
    assert WEB in msg
    assert "2026-05-31" in msg


def test_weekly_summary_empty_is_positive():
    msg = messages.build_weekly_summary(_sel(), WEB)
    assert "ไม่มีจังหวัด" in msg
    assert WEB in msg


def test_weekly_summary_truncates_to_max_list():
    high = [{"name_th": f"จ{i}", "ratio": float(20 - i), "lead": 2} for i in range(15)]
    msg = messages.build_weekly_summary(_sel(high=high), WEB, max_list=12)
    assert "และอีก 3 จังหวัด" in msg


def test_weekly_summary_appends_warnings():
    msg = messages.build_weekly_summary(_sel(warnings=["ข้อมูล MJO ไม่อัปเดต"]), WEB)
    assert "หมายเหตุ" in msg
    assert "MJO" in msg


def test_about_and_welcome_contain_link():
    assert WEB in messages.build_about_message(WEB)
    assert WEB in messages.build_welcome_message(WEB)
