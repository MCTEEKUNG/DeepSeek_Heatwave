from __future__ import annotations

import base64
import hashlib
import hmac

import requests

REPLY_URL = "https://api.line.me/v2/bot/message/reply"
BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"


def verify_signature(channel_secret: str, body: bytes, signature: str) -> bool:
    if not signature or not channel_secret:
        return False
    mac = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def text_message(text: str) -> dict:
    return {"type": "text", "text": text}


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}


def reply(access_token: str, reply_token: str, messages: list, timeout: int = 10):
    payload = {"replyToken": reply_token, "messages": messages}
    return requests.post(REPLY_URL, headers=_headers(access_token), json=payload, timeout=timeout)


def broadcast(access_token: str, messages: list, timeout: int = 10):
    return requests.post(BROADCAST_URL, headers=_headers(access_token),
                         json={"messages": messages}, timeout=timeout)
