from __future__ import annotations

import requests

CREATE_URL = "https://api.line.me/v2/bot/richmenu"
CONTENT_URL = "https://api-data.line.me/v2/bot/richmenu/{rid}/content"
SET_DEFAULT_URL = "https://api.line.me/v2/bot/user/all/richmenu/{rid}"

_W, _H = 2500, 843
_COL = _W // 3  # 833


def build_richmenu_object(website_url: str) -> dict:
    def area(x, width, action):
        return {"bounds": {"x": x, "y": 0, "width": width, "height": _H}, "action": action}

    return {
        "size": {"width": _W, "height": _H},
        "selected": True,
        "name": "heatwave-main",
        "chatBarText": "เมนูคลื่นความร้อน",
        "areas": [
            area(0, _COL, {"type": "postback", "data": "action=weekly",
                           "displayText": "ความเสี่ยงสัปดาห์นี้"}),
            area(_COL, _COL, {"type": "uri", "uri": website_url}),
            area(2 * _COL, _W - 2 * _COL, {"type": "postback", "data": "action=about",
                                           "displayText": "วิธีอ่าน / เกี่ยวกับ"}),
        ],
    }


def create_and_set_default(access_token: str, website_url: str, image_path: str) -> str:
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    r = requests.post(CREATE_URL, headers=headers, json=build_richmenu_object(website_url), timeout=15)
    r.raise_for_status()
    rid = r.json()["richMenuId"]

    ctype = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    with open(image_path, "rb") as f:
        ru = requests.post(CONTENT_URL.format(rid=rid),
                           headers={"Authorization": f"Bearer {access_token}", "Content-Type": ctype},
                           data=f.read(), timeout=30)
    ru.raise_for_status()

    rd = requests.post(SET_DEFAULT_URL.format(rid=rid),
                       headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
    rd.raise_for_status()
    return rid


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m bot.richmenu <image_path>", file=sys.stderr)
        raise SystemExit(2)
    token = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
    web = os.environ.get("WEBSITE_URL", "https://heat-map-frontend.vercel.app/map")
    print("richMenuId =", create_and_set_default(token, web, sys.argv[1]))
