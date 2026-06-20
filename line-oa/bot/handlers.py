from __future__ import annotations

from bot import messages, risk
from bot.line_api import text_message


def _action(event: dict) -> str:
    return (event.get("postback") or {}).get("data", "")


def needs_forecast(event: dict) -> bool:
    return event.get("type") == "postback" and _action(event) == "action=weekly"


def build_reply(event: dict, forecast: dict | None, website_url: str):
    etype = event.get("type")
    if etype == "follow":
        return [text_message(messages.build_welcome_message(website_url))]
    if etype == "postback":
        action = _action(event)
        if action == "action=about":
            return [text_message(messages.build_about_message(website_url))]
        if action == "action=weekly":
            if not forecast:
                return [text_message(
                    f"ขออภัย ขณะนี้ข้อมูลพยากรณ์ยังไม่พร้อม กรุณาดูที่เว็บไซต์: {website_url}")]
            sel = risk.select_risky_provinces(forecast)
            return [text_message(messages.build_weekly_summary(sel, website_url))]
    return None
