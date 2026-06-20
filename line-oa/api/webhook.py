import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# ให้ `import bot.*` ทำงานบน Vercel (เพิ่ม root ของ line-oa เข้า sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import load_config            # noqa: E402
from bot.line_api import verify_signature, reply  # noqa: E402
from bot.handlers import build_reply, needs_forecast  # noqa: E402
from bot.forecast_source import fetch_forecast  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._respond(200, "LINE OA webhook ok")

    def do_POST(self):
        cfg = load_config()
        length = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        signature = self.headers.get("x-line-signature", "")
        if not verify_signature(cfg.channel_secret, body, signature):
            self._respond(401, "invalid signature")
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._respond(400, "bad request")
            return

        events = payload.get("events", [])
        forecast = None
        if any(needs_forecast(e) for e in events):
            try:
                forecast = fetch_forecast(cfg.forecast_url)
            except Exception:
                forecast = None

        for event in events:
            msgs = build_reply(event, forecast, cfg.website_url)
            reply_token = event.get("replyToken")
            if msgs and reply_token:
                try:
                    reply(cfg.channel_access_token, reply_token, msgs)
                except Exception:
                    pass
        self._respond(200, "ok")

    def _respond(self, code: int, text: str):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))
