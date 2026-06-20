from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import line_api, messages, risk      # noqa: E402
from bot.config import load_config            # noqa: E402
from bot.forecast_source import fetch_forecast  # noqa: E402


def load_forecast(cfg, file_path: str | None) -> dict:
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    return fetch_forecast(cfg.forecast_url)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None,
                        help="อ่าน forecast JSON จากไฟล์ local แทน HTTP (ใช้ใน CI)")
    args = parser.parse_args(argv)

    cfg = load_config()
    if not cfg.channel_access_token:
        print("[broadcast] ไม่มี LINE_CHANNEL_ACCESS_TOKEN — ข้าม", file=sys.stderr)
        return 1
    try:
        forecast = load_forecast(cfg, args.file)
    except Exception as e:  # noqa: BLE001
        print(f"[broadcast] ดึง forecast ไม่ได้: {e}", file=sys.stderr)
        return 1

    sel = risk.select_risky_provinces(forecast)
    text = messages.build_weekly_summary(sel, cfg.website_url)
    resp = line_api.broadcast(cfg.channel_access_token, [line_api.text_message(text)])
    print(f"[broadcast] status={resp.status_code} body={resp.text[:200]}")
    return 0 if resp.status_code == 200 else 2


if __name__ == "__main__":
    raise SystemExit(main())
