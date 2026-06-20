import base64
import hashlib
import hmac

from bot import line_api


def _sign(secret: str, body: bytes) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


def test_verify_signature_valid_and_invalid():
    secret, body = "sek", b'{"events":[]}'
    good = _sign(secret, body)
    assert line_api.verify_signature(secret, body, good) is True
    assert line_api.verify_signature(secret, body, "wrong") is False
    assert line_api.verify_signature(secret, body, "") is False
    assert line_api.verify_signature("", body, good) is False  # empty secret -> False


def test_text_message_shape():
    assert line_api.text_message("hi") == {"type": "text", "text": "hi"}


def test_reply_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=10):
        captured["url"], captured["headers"], captured["json"] = url, headers, json
        captured["timeout"] = timeout
        class R: status_code = 200; text = "{}"
        return R()

    monkeypatch.setattr(line_api.requests, "post", fake_post)
    line_api.reply("tok", "rt", [line_api.text_message("hi")])
    assert captured["url"].endswith("/message/reply")
    assert captured["headers"]["Authorization"] == "Bearer tok"
    assert captured["json"]["replyToken"] == "rt"
    assert captured["json"]["messages"][0]["text"] == "hi"
    assert captured["timeout"] == 10


def test_broadcast_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=10):
        captured["url"], captured["json"] = url, json
        captured["timeout"] = timeout
        class R: status_code = 200; text = "{}"
        return R()

    monkeypatch.setattr(line_api.requests, "post", fake_post)
    line_api.broadcast("tok", [line_api.text_message("hi")])
    assert captured["url"].endswith("/message/broadcast")
    assert captured["json"]["messages"][0]["text"] == "hi"
    assert captured["timeout"] == 10
